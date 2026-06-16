import json
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from ai_core import StudyBehaviorDetector
from auth import AuthError, AuthService
from db import Database
from network_utils import (NetworkError, DownloadError, check_url,
                           check_urls_batch, download_image_from_url,
                           test_connectivity)
from visualization import export_alert_chart

RECORD_INTERVAL_SECONDS = 2.0


class LoginFrame(ttk.Frame):
    def __init__(self, master, auth_service, on_login):
        super().__init__(master, padding=24)
        self.auth = auth_service
        self.on_login = on_login
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()

        ttk.Label(self, text="Study Behavior Monitor", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 18)
        )
        ttk.Label(self, text="Username").grid(row=1, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.username_var, width=28).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        ttk.Label(self, text="Password").grid(row=2, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.password_var, show="*", width=28).grid(
            row=2, column=1, sticky="ew", pady=4
        )
        ttk.Button(self, text="Login", command=self.login).grid(
            row=3, column=0, sticky="ew", pady=(12, 0)
        )
        ttk.Button(self, text="Register", command=self.register).grid(
            row=3, column=1, sticky="ew", pady=(12, 0), padx=(8, 0)
        )
        ttk.Label(
            self,
            text="Default admin: admin / admin123",
            foreground="#555",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.columnconfigure(1, weight=1)

    def login(self):
        try:
            user = self.auth.login(self.username_var.get(), self.password_var.get())
        except AuthError as exc:
            messagebox.showerror("Login failed", str(exc))
            return
        self.on_login(user)

    def register(self):
        try:
            user = self.auth.register(self.username_var.get(), self.password_var.get())
        except AuthError as exc:
            messagebox.showerror("Register failed", str(exc))
            return
        messagebox.showinfo("Registered", f"User {user['username']} has been created.")


class MainFrame(ttk.Frame):
    def __init__(self, master, database, auth_service, user):
        super().__init__(master, padding=18)
        self.db = database
        self.auth = auth_service
        self.user = user
        self.detector = StudyBehaviorDetector()
        self.status_var = tk.StringVar(value="Ready")
        self.running = False
        self.current_session_id = None
        self._last_record_time = {}

        title = f"Logged in as {user['username']} ({user['role']})"
        ttk.Label(self, text=title, font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        ttk.Button(self, text="Detect Image", command=self.detect_image).grid(
            row=1, column=0, sticky="ew", padx=(0, 8), pady=4
        )
        ttk.Button(self, text="Detect from URL", command=self.detect_from_url).grid(
            row=1, column=1, sticky="ew", padx=(0, 8), pady=4
        )
        ttk.Button(self, text="Start Camera", command=self.start_camera).grid(
            row=1, column=2, sticky="ew", pady=4
        )
        ttk.Button(self, text="Check Model URL", command=self.check_model_url).grid(
            row=2, column=0, sticky="ew", padx=(0, 8), pady=4
        )
        ttk.Button(self, text="Batch URL Check", command=self.batch_url_check).grid(
            row=2, column=1, sticky="ew", padx=(0, 8), pady=4
        )
        ttk.Button(self, text="Network Test", command=self.network_test).grid(
            row=2, column=2, sticky="ew", pady=4
        )
        ttk.Button(self, text="Detection Records", command=self.show_records).grid(
            row=3, column=0, sticky="ew", padx=(0, 8), pady=4
        )
        ttk.Button(self, text="Operation Logs", command=self.show_logs).grid(
            row=3, column=1, sticky="ew", padx=(0, 8), pady=4
        )
        ttk.Button(self, text="Export Chart", command=self.export_chart).grid(
            row=3, column=2, sticky="ew", pady=4
        )
        ttk.Button(self, text="Change Password", command=self.change_password).grid(
            row=4, column=0, sticky="ew", padx=(0, 8), pady=4
        )
        if self.auth.is_admin(self.user):
            ttk.Button(self, text="Manage Users", command=self.manage_users).grid(
                row=4, column=1, sticky="ew", padx=(0, 8), pady=4
            )
            ttk.Button(self, text="Database Admin", command=self.open_db_admin).grid(
                row=4, column=2, sticky="ew", padx=(0, 8), pady=4
            )

        ttk.Label(self, textvariable=self.status_var, foreground="#444").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(16, 0)
        )
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=200)
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.progress.grid_remove()
        for column in range(3):
            self.columnconfigure(column, weight=1)

    def _record_behavior_events(self, events, source_type, source_path, output_path, session_id):
        for ev in events:
            self.db.add_behavior_record(
                user_id=self.user["id"],
                behavior_type=ev["label"],
                confidence=ev["confidence"],
                session_id=session_id,
                is_alert=ev["alert"],
                alert_reason=ev["reason"],
                source_type=source_type,
                source_path=source_path,
                output_image_path=output_path,
                extra_info={"bbox": ev["bbox"]}
            )

    def _should_record_event(self, session_id, label, is_alert):
        if is_alert:
            return True
        key = (session_id, label)
        now = datetime.now().timestamp()
        last = self._last_record_time.get(key, 0)
        if now - last >= RECORD_INTERVAL_SECONDS:
            self._last_record_time[key] = now
            return True
        return False

    def _end_session_with_stats(self, session_id, summary):
        alert_labels = summary.get("alert_labels", [])
        stats = {
            "total": len(alert_labels),
            "phone": alert_labels.count("phone"),
            "sleep": alert_labels.count("sleep"),
            "eat": alert_labels.count("eat"),
        }
        self.db.end_study_session(session_id, alert_stats=stats)

    def detect_image(self):
        path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._run_task(lambda: self._detect_image_worker(path), "Detecting image...")

    def _detect_image_worker(self, path):
        session_id = self.db.start_study_session(self.user["id"], f"Image: {Path(path).name}")
        try:
            output_path, events, summary = self.detector.predict_image(path)
            self._record_behavior_events(events, "image", str(path), str(output_path), session_id)
            self._end_session_with_stats(session_id, summary)
            alerts = summary.get("alert_labels", [])
            self.db.record_detection(
                self.user["id"], str(path), summary, alerts, output_path=output_path
            )
            self.db.log_operation(
                self.user["id"], "detect_image", f"{Path(path).name} -> {output_path.name}"
            )
            message = (
                f"Output: {output_path}\n"
                f"Events: {len(events)}\n"
                f"Alerts: {', '.join(alerts) if alerts else 'none'}"
            )
            self.after(0, lambda: messagebox.showinfo("Detection finished", message))
        except Exception:
            self.db.end_study_session(session_id)
            raise

    def detect_from_url(self):
        url = simpledialog.askstring(
            "Detect from URL",
            "Input an image URL:",
            parent=self,
        )
        if not url:
            return
        self._run_task(lambda: self._detect_url_worker(url), "Downloading & detecting...")

    def _detect_url_worker(self, url):
        try:
            local_path = download_image_from_url(url)
        except DownloadError as exc:
            self.after(0, lambda: messagebox.showerror("Download failed", str(exc)))
            return
        session_id = self.db.start_study_session(self.user["id"], f"URL: {url[:50]}")
        try:
            output_path, events, summary = self.detector.predict_image(str(local_path))
            self._record_behavior_events(events, "image", url, str(output_path), session_id)
            self._end_session_with_stats(session_id, summary)
            alerts = summary.get("alert_labels", [])
            self.db.record_detection(
                self.user["id"], url, summary, alerts, output_path=output_path
            )
            self.db.log_operation(
                self.user["id"], "detect_url", f"{url} -> {output_path}"
            )
            message = (
                f"Source URL: {url}\n"
                f"Output: {output_path}\n"
                f"Events: {len(events)}\n"
                f"Alerts: {', '.join(alerts) if alerts else 'none'}"
            )
            self.after(0, lambda: messagebox.showinfo("Detection finished", message))
        except Exception:
            self.db.end_study_session(session_id)
            raise

    def start_camera(self):
        if self.running:
            messagebox.showinfo("Camera", "Camera detection is already running.")
            return
        source = simpledialog.askstring(
            "Camera source",
            "Input camera index or video path:",
            initialvalue="0",
            parent=self,
        )
        if source is None:
            return
        try:
            source_value = int(source)
        except ValueError:
            source_value = source
        session_name = f"Camera {datetime.now().strftime('%H:%M:%S')}"
        self.current_session_id = self.db.start_study_session(self.user["id"], session_name)
        self._last_record_time.clear()
        self._run_task(lambda: self._camera_worker(source_value), "Camera running...")

    def _camera_worker(self, source):
        self.running = True
        self.db.log_operation(self.user["id"], "start_camera", str(source))

        def on_frame(annotated, events, summary):
            if events:
                for ev in events:
                    if self._should_record_event(self.current_session_id, ev["label"], ev["alert"]):
                        self.db.add_behavior_record(
                            user_id=self.user["id"],
                            behavior_type=ev["label"],
                            confidence=ev["confidence"],
                            session_id=self.current_session_id,
                            is_alert=ev["alert"],
                            alert_reason=ev["reason"],
                            source_type="camera",
                            source_path=str(source),
                            extra_info={"bbox": ev["bbox"]}
                        )
            if summary["alert_labels"]:
                self.status_var.set(f"Alert: {', '.join(summary['alert_labels'])}")

        try:
            self.detector.run_camera(source, on_frame=on_frame)
        finally:
            records = self.db.get_behavior_records(session_id=self.current_session_id)
            alert_counts = {"total": 0, "phone": 0, "sleep": 0, "eat": 0}
            for r in records:
                if r["is_alert"]:
                    alert_counts["total"] += 1
                    behavior = r["behavior_type"]
                    if behavior in alert_counts:
                        alert_counts[behavior] += 1
            self.db.end_study_session(self.current_session_id, alert_stats=alert_counts)
            self.db.log_operation(self.user["id"], "stop_camera", str(source))
            self.current_session_id = None
            self.running = False
            self.after(0, lambda: self.status_var.set("Camera stopped"))

    def check_model_url(self):
        url = simpledialog.askstring(
            "Model URL",
            "Input a model or dataset URL to check:",
            parent=self,
        )
        if not url:
            return
        self._run_task(lambda: self._check_url_worker(url), "Checking network...")

    def _check_url_worker(self, url):
        try:
            result = check_url(url)
            detail = json.dumps(result, ensure_ascii=False)
            self.db.upsert_model_resource("remote_check", url, None, "reachable")
            self.db.log_operation(self.user["id"], "check_url", detail)
            self.after(0, lambda: messagebox.showinfo("Network", detail))
        except NetworkError as exc:
            self.db.upsert_model_resource("remote_check", url, None, "failed")
            self.db.log_operation(self.user["id"], "check_url_failed", str(exc))
            self.after(0, lambda: messagebox.showerror("Network failed", str(exc)))

    def network_test(self):
        self._run_task(self._network_test_worker, "Testing connectivity...")

    def _network_test_worker(self):
        result = test_connectivity()
        self.db.log_operation(self.user["id"], "network_test", json.dumps(result))
        if result["reachable"]:
            msg = (f"Network reachable\n"
                   f"Latency: {result['latency_ms']} ms\n"
                   f"Resolved IP: {result['resolved_ip']}")
            self.after(0, lambda: messagebox.showinfo("Network Test", msg))
        else:
            self.after(0, lambda: messagebox.showerror(
                "Network Test", f"Not reachable: {result.get('error', 'unknown')}"))

    def batch_url_check(self):
        text = simpledialog.askstring(
            "Batch URL Check",
            "Input URLs to check (one per line or comma-separated):",
            parent=self,
        )
        if not text:
            return
        urls = [u.strip() for line in text.splitlines() for u in line.split(",")
                if u.strip()]
        if not urls:
            return
        self._run_task(lambda: self._batch_url_worker(urls), "Checking URLs...")

    def _batch_url_worker(self, urls):
        results = check_urls_batch(urls)
        self.db.log_operation(
            self.user["id"], "batch_url_check", f"Checked {len(urls)} URLs"
        )
        lines = []
        for url, result in results.items():
            status = "OK" if result.get("ok") else "FAIL"
            detail = result.get("error", result.get("status", ""))
            lines.append(f"[{status}] {url}  {detail}")
        self.after(0, lambda: self._show_text("Batch URL Results", "\n".join(lines)))

    def show_records(self):
        records = self.db.list_detection_records()
        lines = []
        for record in records:
            lines.append(
                f"#{record['id']} {record['created_at']} {record['username'] or '-'}\n"
                f"source: {record['source']}\n"
                f"alerts: {record['alerts_json']}\n"
                f"output: {record['output_path'] or '-'}\n"
            )
        self._show_text("Detection Records", "\n".join(lines) or "No records.")

    def show_logs(self):
        logs = self.db.list_operation_logs()
        lines = [
            f"#{item['id']} {item['created_at']} {item['username'] or '-'} "
            f"{item['action']} {item['detail'] or ''}"
            for item in logs
        ]
        self._show_text("Operation Logs", "\n".join(lines) or "No logs.")

    def export_chart(self):
        target = filedialog.asksaveasfilename(
            title="Save chart",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not target:
            return
        try:
            output = export_alert_chart(self.db.list_detection_records(), target)
        except RuntimeError as exc:
            messagebox.showerror("Chart failed", str(exc))
            return
        self.db.log_operation(self.user["id"], "export_chart", str(output))
        messagebox.showinfo("Chart exported", str(output))

    def change_password(self):
        old_password = simpledialog.askstring(
            "Old password", "Input old password:", parent=self, show="*"
        )
        if old_password is None:
            return
        new_password = simpledialog.askstring(
            "New password", "Input new password:", parent=self, show="*"
        )
        if not new_password:
            return
        try:
            self.auth.change_password(
                self.user["username"], old_password, new_password, operator=self.user
            )
        except AuthError as exc:
            messagebox.showerror("Change failed", str(exc))
            return
        messagebox.showinfo("Password", "Password changed.")

    def manage_users(self):
        if not self.auth.is_admin(self.user):
            messagebox.showerror("Forbidden", "Admin permission required.")
            return
        win = tk.Toplevel(self)
        win.title("User Management")
        win.geometry("850x500")
        win.minsize(700, 400)
        user_mgmt = UserManagementFrame(win, self.db, self.auth, self.user)
        user_mgmt.pack(fill="both", expand=True)

    def open_db_admin(self):
        win = tk.Toplevel(self)
        win.title("Database Administration")
        win.geometry("900x600")
        win.minsize(800, 400)
        db_admin = DatabaseAdminFrame(win, self.db, self.user)
        db_admin.pack(fill="both", expand=True)

    def _run_task(self, target, status):
        self.status_var.set(status)
        self.progress.grid()
        self.progress.start(10)

        def runner():
            try:
                target()
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
            finally:
                self.after(0, lambda: self._task_done(status))

        threading.Thread(target=runner, daemon=True).start()

    def _task_done(self, status):
        self.progress.stop()
        self.progress.grid_remove()
        self.status_var.set("Ready")

    def _show_text(self, title, text):
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("760x480")
        text_widget = tk.Text(window, wrap="word")
        text_widget.insert("1.0", text)
        text_widget.configure(state="disabled")
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)


class UserManagementFrame(ttk.Frame):
    def __init__(self, master, database, auth_service, admin_user):
        super().__init__(master, padding=10)
        self.db = database
        self.auth = auth_service
        self.admin_user = admin_user
        self.tree = None
        self.comboboxes = {}
        self._build_ui()
        self.refresh_users()

    def _build_ui(self):
        create_frame = ttk.LabelFrame(self, text="Create New User", padding=8)
        create_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(create_frame, text="Username:").grid(row=0, column=0, padx=5, sticky="w")
        self.new_username = tk.StringVar()
        ttk.Entry(create_frame, textvariable=self.new_username, width=15).grid(row=0, column=1, padx=5)

        ttk.Label(create_frame, text="Password:").grid(row=0, column=2, padx=5, sticky="w")
        self.new_password = tk.StringVar()
        ttk.Entry(create_frame, textvariable=self.new_password, width=15, show="*").grid(row=0, column=3, padx=5)

        ttk.Label(create_frame, text="Role:").grid(row=0, column=4, padx=5, sticky="w")
        self.new_role = tk.StringVar(value="user")
        role_combo = ttk.Combobox(create_frame, textvariable=self.new_role, values=["user", "admin"], width=8, state="readonly")
        role_combo.grid(row=0, column=5, padx=5)

        ttk.Button(create_frame, text="Create User", command=self.create_user).grid(row=0, column=6, padx=10)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True)

        columns = ("ID", "Username", "Role", "Created At", "Action")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        self.tree.heading("ID", text="ID")
        self.tree.heading("Username", text="Username")
        self.tree.heading("Role", text="Role")
        self.tree.heading("Created At", text="Created At")
        self.tree.heading("Action", text="Action")
        self.tree.column("ID", width=50, anchor="center")
        self.tree.column("Username", width=120)
        self.tree.column("Role", width=80, anchor="center")
        self.tree.column("Created At", width=160)
        self.tree.column("Action", width=150, anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_frame, text="Refresh", command=self.refresh_users).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Close", command=self.master.destroy).pack(side="right", padx=2)

    def refresh_users(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.comboboxes.clear()

        try:
            users = self.auth.list_users(self.admin_user)
        except AuthError as e:
            messagebox.showerror("Error", str(e))
            return

        for user in users:
            item_id = str(user["id"])
            self.tree.insert("", "end", iid=item_id,
                             values=(user["id"], user["username"], user["role"], user["created_at"], "operate"))

    def on_tree_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.tree.identify_column(event.x)
        columns = self.tree["columns"]
        if column == f"#{len(columns)}":
            item = self.tree.identify_row(event.y)
            if not item:
                return
            values = self.tree.item(item, "values")
            if not values:
                return
            user_id = values[0]
            username = values[1]
            current_role = values[2]
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Reset Password", command=lambda: self.reset_password(user_id, username))
            menu.add_command(label="Delete User", command=lambda: self.delete_user(user_id, username))
            role_menu = tk.Menu(menu, tearoff=0)
            role_menu.add_command(label="Set as user", command=lambda: self.change_role(user_id, username, "user"))
            role_menu.add_command(label="Set as admin", command=lambda: self.change_role(user_id, username, "admin"))
            menu.add_cascade(label="Change Role", menu=role_menu)
            menu.post(event.x_root, event.y_root)

    def reset_password(self, user_id, username):
        new_password = simpledialog.askstring(
            "Reset Password",
            f"Enter new password for user '{username}':",
            parent=self.master,
            show="*"
        )
        if not new_password:
            return
        try:
            self.auth.reset_password(username, new_password, self.admin_user)
        except AuthError as e:
            messagebox.showerror("Error", str(e))
            return
        messagebox.showinfo("Success", f"Password for '{username}' has been reset.")
        self.db.log_operation(self.admin_user["id"], "reset_password", f"Reset password for {username}")

    def delete_user(self, user_id, username):
        if username == "admin":
            messagebox.showerror("Error", "Cannot delete the default admin account.")
            return
        if username == self.admin_user["username"]:
            messagebox.showerror("Error", "You cannot delete your own account.")
            return
        if not messagebox.askyesno("Confirm Delete", f"Delete user '{username}'?"):
            return
        try:
            self.auth.delete_user(username, self.admin_user)
        except AuthError as e:
            messagebox.showerror("Error", str(e))
            return
        messagebox.showinfo("Success", f"User '{username}' deleted.")
        self.db.log_operation(self.admin_user["id"], "delete_user", f"Deleted {username}")
        self.refresh_users()

    def change_role(self, user_id, username, new_role):
        if username == "admin" and new_role != "admin":
            messagebox.showerror("Error", "Cannot demote default admin.")
            return
        try:
            self.auth.change_role(username, new_role, self.admin_user)
        except AuthError as e:
            messagebox.showerror("Error", str(e))
            return
        messagebox.showinfo("Success", f"Role for '{username}' changed to {new_role}.")
        self.db.log_operation(self.admin_user["id"], "change_role", f"Changed {username} to {new_role}")
        self.refresh_users()

    def create_user(self):
        username = self.new_username.get().strip()
        password = self.new_password.get().strip()
        role = self.new_role.get()
        if not username or not password:
            messagebox.showerror("Error", "Username and password cannot be empty.")
            return
        try:
            self.auth.register(username, password, role, self.admin_user)
        except AuthError as e:
            messagebox.showerror("Error", str(e))
            return
        messagebox.showinfo("Success", f"User '{username}' created with role '{role}'.")
        self.db.log_operation(self.admin_user["id"], "create_user", f"Created {username} as {role}")
        self.new_username.set("")
        self.new_password.set("")
        self.new_role.set("user")
        self.refresh_users()


class DatabaseAdminFrame(ttk.Frame):
    ALLOW_DELETE = {
        "users",
        "detection_records",
        "study_sessions",
        "study_behavior_records",
        "model_resources",
        "misclassified_samples",
        "data_reflux_log",
        "training_cycles",
    }
    EXCLUDE_COLUMNS = {
        "users": ["password_hash", "salt"],
    }
    PK_COLUMN = "id"

    def __init__(self, master, database, user):
        super().__init__(master, padding=10)
        self.db = database
        self.user = user
        self.tables = self._get_table_list()
        self.treeviews = {}
        self.current_table = None

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Button(toolbar, text="Refresh Current", command=self.refresh_current).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Delete Selected", command=self.delete_selected).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Close", command=master.destroy).pack(side="right", padx=2)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        for table in self.tables:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=table)

            tree_frame = ttk.Frame(frame)
            tree_frame.pack(fill="both", expand=True)

            tree = ttk.Treeview(tree_frame, show="headings")
            vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            tree_frame.grid_rowconfigure(0, weight=1)
            tree_frame.grid_columnconfigure(0, weight=1)

            self.treeviews[table] = tree
            self._populate_treeview(table, tree)

        for table, tree in self.treeviews.items():
            tree.bind("<Button-3>", lambda event, t=table, tr=tree: self._show_context_menu(event, t, tr))

        if self.tables:
            self.current_table = self.tables[0]

    def _get_table_list(self):
        return [
            "users",
            "operation_logs",
            "detection_records",
            "study_sessions",
            "study_behavior_records",
            "model_resources",
            "misclassified_samples",
            "data_reflux_log",
            "training_cycles",
        ]

    def _populate_treeview(self, table, tree):
        for item in tree.get_children():
            tree.delete(item)

        columns = self._get_columns(table)
        if not columns:
            return
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, minwidth=60, anchor="w")

        try:
            with self.db.connect() as conn:
                cursor = conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 1000")
                rows = cursor.fetchall()
        except Exception as e:
            messagebox.showerror("DB Error", f"Failed to fetch {table}: {e}")
            return

        for row in rows:
            values = [row[col] for col in columns]
            iid = str(row[self.PK_COLUMN]) if self.PK_COLUMN in columns else ""
            tree.insert("", "end", iid=iid, values=values)

    def _get_columns(self, table):
        try:
            with self.db.connect() as conn:
                cursor = conn.execute(f"PRAGMA table_info({table})")
                cols = [row["name"] for row in cursor.fetchall()]
        except Exception:
            return []
        exclude = self.EXCLUDE_COLUMNS.get(table, [])
        return [c for c in cols if c not in exclude]

    def _on_tab_changed(self, event):
        tab = self.notebook.select()
        if tab:
            index = self.notebook.index(tab)
            if 0 <= index < len(self.tables):
                self.current_table = self.tables[index]

    def refresh_current(self):
        if not self.current_table:
            return
        tree = self.treeviews.get(self.current_table)
        if tree:
            self._populate_treeview(self.current_table, tree)

    def delete_selected(self):
        if not self.current_table:
            messagebox.showinfo("Info", "No table selected.")
            return
        if self.current_table not in self.ALLOW_DELETE:
            messagebox.showwarning("Forbidden", f"Deletion not allowed for table '{self.current_table}'.")
            return

        tree = self.treeviews.get(self.current_table)
        if not tree:
            return
        selected = tree.selection()
        if not selected:
            messagebox.showinfo("Info", "No row selected.")
            return

        pk_values = []
        for item in selected:
            values = tree.item(item, "values")
            if values:
                pk = values[0]
                if pk:
                    pk_values.append(pk)

        if not pk_values:
            return

        if not messagebox.askyesno(
            "Confirm Delete",
            f"Delete {len(pk_values)} record(s) from '{self.current_table}'?"
        ):
            return

        if self.current_table == "users":
            for pk in pk_values:
                try:
                    with self.db.connect() as conn:
                        row = conn.execute("SELECT username FROM users WHERE id = ?", (pk,)).fetchone()
                        if row and row["username"] == "admin":
                            messagebox.showerror("Error", "Cannot delete the default admin account.")
                            return
                except Exception:
                    pass

        try:
            with self.db.connect() as conn:
                placeholders = ",".join("?" * len(pk_values))
                sql = f"DELETE FROM {self.current_table} WHERE id IN ({placeholders})"
                conn.execute(sql, pk_values)
        except Exception as e:
            messagebox.showerror("Delete Error", str(e))
            return

        self.refresh_current()
        messagebox.showinfo("Success", f"Deleted {len(pk_values)} record(s).")

    def _show_context_menu(self, event, table, tree):
        item = tree.identify_row(event.y)
        if not item:
            return
        tree.selection_set(item)
        if table == "study_behavior_records":
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Save as Misclassified Sample",
                             command=lambda: self._save_as_misclassified(tree, item))
            menu.post(event.x_root, event.y_root)

    def _save_as_misclassified(self, tree, item):
        values = tree.item(item, "values")
        if not values:
            messagebox.showerror("Error", "Unable to get record data.")
            return
        try:
            record_id = int(values[0])
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Unable to get record ID.")
            return

        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM study_behavior_records WHERE id = ?", (record_id,)).fetchone()
        if not row:
            messagebox.showerror("Error", "Record not found in database.")
            return

        record = dict(row)
        img_path = record.get("output_image_path") or record.get("image_path") or record.get("source_path")
        if not img_path:
            if not messagebox.askyesno("Warning", "No image path associated. Save text only?"):
                return

        predictions = {
            "label": record["behavior_type"],
            "confidence": record["confidence"],
            "bbox": json.loads(record.get("extra_info", "{}")).get("bbox", [])
        }

        try:
            sample_id = self.db.insert_misclassified_sample(
                image_path=img_path or "",
                original_predictions=predictions,
                user_id=self.user["id"]
            )
            self.db.log_operation(
                self.user["id"],
                "collect_misclassified",
                f"Saved behavior record #{record_id} as misclassified sample #{sample_id}"
            )
            messagebox.showinfo("Success", f"Saved as misclassified sample (ID: {sample_id})")
            self.refresh_current()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")


class StudyMonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Study Behavior Monitor")
        self.geometry("720x440")
        self.minsize(640, 400)

        self.db = Database()
        self.auth = AuthService(self.db)
        self.auth.ensure_default_admin()
        self.current_frame = None
        self.show_login()

    def show_login(self):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = LoginFrame(self, self.auth, self.show_main)
        self.current_frame.pack(fill="both", expand=True)

    def show_main(self, user):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = MainFrame(self, self.db, self.auth, user)
        self.current_frame.pack(fill="both", expand=True)


def main():
    app = StudyMonitorApp()
    app.mainloop()