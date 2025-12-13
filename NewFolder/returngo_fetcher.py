import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests
import json
import threading
from collections import defaultdict
from datetime import datetime, timedelta

# --- Optional: Calendar Widget ---
try:
    from tkcalendar import DateEntry
    HAS_CALENDAR = True
except Exception:
    DateEntry = None
    HAS_CALENDAR = False

# --- Configuration ---
API_URL_LIST = "https://api.returngo.ai/rmas"
API_URL_DETAIL = "https://api.returngo.ai/rma"
API_KEY = "G0dxSEprxa5iOAIsK5X6g7fCDdPl09q2882osmWD"

SHOP_LIST = [
    "diesel-dev-south-africa.myshopify.com",
    "hurley-dev-south-africa.myshopify.com",
    "jeep-apparel-dev-south-africa.myshopify.com",
    "reebok-dev-south-africa.myshopify.com",
    "superdry-dev-south-africa.myshopify.com"
]

SHORT_SHOP_NAMES = [s.replace(".myshopify.com", "") for s in SHOP_LIST]
DROPDOWN_SHOPS = ["ALL STORES"] + SHOP_LIST

STATUS_OPTIONS = [
    "Pending", "Approved", "Received", "Validated",
    "Done", "Rejected", "Canceled"
]

class ReturnGoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ReturnGO RMA Analytics (Threaded & Logging)")
        self.root.geometry("1400x800") # Taller for log window

        # --- Top Control Panel ---
        control_frame = ttk.LabelFrame(root, text="Search Criteria", padding="10")
        control_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(control_frame, text="Select Shop:").grid(row=0, column=0, sticky="w", padx=5)
        self.shop_var = tk.StringVar(value=DROPDOWN_SHOPS[0])
        self.shop_dropdown = ttk.Combobox(control_frame, textvariable=self.shop_var, values=DROPDOWN_SHOPS, width=35, state="readonly")
        self.shop_dropdown.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(control_frame, text="Status:").grid(row=0, column=2, sticky="w", padx=5)
        self.status_var = tk.StringVar(value=STATUS_OPTIONS[1]) 
        self.status_dropdown = ttk.Combobox(control_frame, textvariable=self.status_var, values=STATUS_OPTIONS, state="readonly")
        self.status_dropdown.grid(row=0, column=3, sticky="w", padx=5)

        ttk.Label(control_frame, text="From Date:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        default_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        if HAS_CALENDAR:
            self.start_date_entry = DateEntry(control_frame, date_pattern="yyyy-MM-dd")
            self.start_date_entry.set_date(default_start)
        else:
            self.start_date_entry = ttk.Entry(control_frame)
            self.start_date_entry.insert(0, default_start)
        self.start_date_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(control_frame, text="To Date:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        default_end = datetime.now().strftime("%Y-%m-%d")
        if HAS_CALENDAR:
            self.end_date_entry = DateEntry(control_frame, date_pattern="yyyy-MM-dd")
            self.end_date_entry.set_date(default_end)
        else:
            self.end_date_entry = ttk.Entry(control_frame)
            self.end_date_entry.insert(0, default_end)
        self.end_date_entry.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=10)

        self.fetch_btn = ttk.Button(btn_frame, text="Fetch Data", command=self.start_fetch_thread)
        self.fetch_btn.pack(side="left", padx=5)

        self.copy_btn = ttk.Button(btn_frame, text="Copy Current Tab", command=self.copy_to_clipboard)
        self.copy_btn.pack(side="left", padx=5)

        # --- LIVE LOG WINDOW ---
        log_labelframe = ttk.LabelFrame(root, text="Live Logs (Debugging)", padding="5")
        log_labelframe.pack(fill="x", padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_labelframe, height=8, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        # --- TABS ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_detail = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_detail, text="  Detailed List  ")
        self.setup_detail_tab()

        self.tab_summary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_summary, text="  Daily Breakdown (Pivot)  ")
        self.setup_summary_tab()

        self.all_fetched_rows = []

    def log(self, message):
        """Updates the log window safely from any thread."""
        def _update():
            self.log_text.config(state='normal')
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END) # Auto-scroll
            self.log_text.config(state='disabled')
        self.root.after(0, _update)

    def setup_detail_tab(self):
        table_frame = ttk.Frame(self.tab_detail)
        table_frame.pack(fill="both", expand=True)
        scroll_y = ttk.Scrollbar(table_frame)
        scroll_y.pack(side="right", fill="y")
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")
        scroll_x.pack(side="bottom", fill="x")

        columns = ("shop", "rma_id", "order_name", "status", "created_date", "action_date", "action_by", "trigger")
        self.detail_tree = ttk.Treeview(table_frame, columns=columns, show="headings", 
                                        yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        
        for col in columns:
            self.detail_tree.heading(col, text=col.replace("_", " ").title().replace("Rma", "RMA"))
            self.detail_tree.column(col, width=120)

        self.detail_tree.pack(fill="both", expand=True)
        scroll_y.config(command=self.detail_tree.yview)
        scroll_x.config(command=self.detail_tree.xview)

    def setup_summary_tab(self):
        table_frame = ttk.Frame(self.tab_summary)
        table_frame.pack(fill="both", expand=True)
        scroll_y = ttk.Scrollbar(table_frame)
        scroll_y.pack(side="right", fill="y")
        scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")
        scroll_x.pack(side="bottom", fill="x")

        self.pivot_columns = ["date"] + SHORT_SHOP_NAMES + ["total"]
        self.summary_tree = ttk.Treeview(table_frame, columns=self.pivot_columns, show="headings", 
                                         yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        self.summary_tree.heading("date", text="Date")
        self.summary_tree.column("date", width=100, anchor="center")
        for shop in SHORT_SHOP_NAMES:
            display = shop.split("-dev-")[0].capitalize() if "-dev-" in shop else shop[:10]
            self.summary_tree.heading(shop, text=display)
            self.summary_tree.column(shop, width=80, anchor="center")
        self.summary_tree.heading("total", text="Total")
        self.summary_tree.column("total", width=60, anchor="e")

        self.summary_tree.pack(fill="both", expand=True)
        scroll_y.config(command=self.summary_tree.yview)
        scroll_x.config(command=self.summary_tree.xview)

    def get_text_from_date_widget(self, widget):
        if HAS_CALENDAR and isinstance(widget, DateEntry):
            return widget.get_date().strftime("%Y-%m-%d")
        return widget.get()

    def get_api_params(self):
        status = self.status_var.get()
        start_date_str = self.get_text_from_date_widget(self.start_date_entry)
        end_date_str = self.get_text_from_date_widget(self.end_date_entry)
        try:
            start_iso = f"gte:{start_date_str}T00:00:00"
            end_iso = f"lte:{end_date_str}T23:59:59"
        except Exception:
            return None
        return {"pagesize": 50, "status": status, "rma_created_at": [start_iso, end_iso]}, start_date_str, end_date_str

    def fetch_deep_details(self, rma_id, shop_name):
        url = f"{API_URL_DETAIL}/{rma_id}"
        headers = { "accept": "application/json", "x-shop-name": shop_name, "x-api-key": API_KEY }
        try:
            res = requests.get(url, headers=headers, timeout=10) # Increased timeout
            if res.status_code == 200:
                return res.json()
            else:
                self.log(f"Detail fetch failed for {rma_id}: Status {res.status_code}")
        except Exception as e:
            self.log(f"Detail fetch error for {rma_id}: {e}")
        return {}

    def fetch_single_shop(self, shop_name, params, start_date_filter, end_date_filter):
        headers = { "accept": "application/json", "x-shop-name": shop_name, "x-api-key": API_KEY }
        rows = []
        try:
            self.log(f"Querying list from: {shop_name}...")
            response = requests.get(API_URL_LIST, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                rmas = data.get("rmas", []) if isinstance(data, dict) else []
                self.log(f"Found {len(rmas)} raw records for {shop_name}.")

                for rma in rmas:
                    rma_id = rma.get("rmaId", rma.get("id"))
                    order_name = rma.get("order_name") or "N/A"
                    status = rma.get("status", "N/A")
                    created_date_raw = rma.get("createdAt")
                    
                    # --- Date Filter Check ---
                    if created_date_raw:
                        row_date_str = created_date_raw[:10]
                        if row_date_str < start_date_filter or row_date_str > end_date_filter:
                            continue # Skip

                    action_date = "-"
                    action_by_user = "-"

                    # --- DEEP FETCH ---
                    # We do a deep scan for statuses that involve a manual user action we want to track.
                    if status in ["Approved", "Done", "Rejected"]:
                        self.log(f"  > Deep scanning '{status}' RMA: {rma_id} ({order_name})")
                        full_details = self.fetch_deep_details(rma_id, shop_name)
                        comments = full_details.get("comments", [])
                        
                        found_user = False
                        for comment in comments:
                            html = str(comment.get("htmlText", "")).lower()
                            c_user = comment.get("triggeredBy")
                            c_date = comment.get("datetime")

                            # Logic for 'Approved'
                            if status == "Approved" and "approved" in html and ("created" in html or "request" in html):
                                if c_user and c_user != "ReturnGO API":
                                    action_by_user = c_user
                                    found_user = True
                                if c_date:
                                    action_date = c_date[:10]
                                    break # Stop searching comments
                            
                            # Logic for 'Done'
                            elif status == "Done" and "rma status changed to done" in html:
                                if c_user and c_user != "ReturnGO API":
                                    action_by_user = c_user
                                    found_user = True
                                # Also grab the date for the 'Done' action
                                if c_date:
                                    action_date = c_date[:10]
                                    found_user = True
                                    break # Stop searching comments
                            
                            # Logic for 'Rejected' could be added here similarly
                        
                        if found_user:
                            self.log(f"    -> Found User: {action_by_user}")
                        else:
                            self.log(f"    -> No explicit user found in comments for this action.")
                    
                    # Clean up
                    if created_date_raw: created_date_raw = created_date_raw.replace("T", " ")[:19]

                    rows.append({
                        "shop": shop_name.replace(".myshopify.com", ""),
                        "rma_id": rma_id,
                        "order_name": order_name,
                        "status": status,
                        "created_date": created_date_raw or "N/A",
                        "action_date": action_date,
                        "action_by": action_by_user,
                        "trigger": rma.get("triggeredVia", "-")
                    })
            else:
                self.log(f"Error fetching list: {response.status_code}")
        except Exception as e:
            self.log(f"Error for {shop_name}: {e}")
        return rows

    def start_fetch_thread(self):
        """Starts the background thread."""
        self.fetch_btn.config(state="disabled", text="Running...")
        self.copy_btn.config(state="disabled")
        
        # Clear logs
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        
        # Start Thread
        thread = threading.Thread(target=self.run_process_background)
        thread.daemon = True # Ensures thread dies if app closes
        thread.start()

    def run_process_background(self):
        """The actual work happening in the background."""
        # 1. Clear Tables (Must be done via main thread really, but usually ok here, or use root.after)
        self.root.after(0, self.clear_tables)

        self.all_fetched_rows = []
        selected_option = self.shop_var.get()
        params, start_date, end_date = self.get_api_params()
        
        if not params:
            self.log("Invalid Date Parameters.")
            self.root.after(0, self.finish_fetching)
            return

        shops = SHOP_LIST if selected_option == "ALL STORES" else [selected_option]
        
        for shop in shops:
            rows = self.fetch_single_shop(shop, params, start_date, end_date)
            self.all_fetched_rows.extend(rows)

        # Update UI when done
        self.root.after(0, self.populate_ui_and_finish)

    def clear_tables(self):
        for i in self.detail_tree.get_children(): self.detail_tree.delete(i)
        for i in self.summary_tree.get_children(): self.summary_tree.delete(i)

    def populate_ui_and_finish(self):
        # Populate Detail
        for row in self.all_fetched_rows:
            self.detail_tree.insert("", "end", values=(
                row["shop"], row["rma_id"], row["order_name"], row["status"],
                row["created_date"], row["action_date"], row["action_by"], row["trigger"]
            ))
        
        self.populate_pivot_summary()
        self.log("Done! Process Complete.")
        self.fetch_btn.config(state="normal", text="Fetch Data")
        self.copy_btn.config(state="normal")
        messagebox.showinfo("Done", f"Fetched {len(self.all_fetched_rows)} records.")

    def populate_pivot_summary(self):
        matrix = defaultdict(lambda: defaultdict(int))
        all_dates = set()
        for row in self.all_fetched_rows:
            d_str = row["action_date"] if row["action_date"] != "-" else row["created_date"][:10]
            if not d_str: continue
            matrix[d_str][row["shop"]] += 1
            all_dates.add(d_str)

        for d in sorted(list(all_dates), reverse=True):
            vals = [d]
            tot = 0
            for shop in SHORT_SHOP_NAMES:
                c = matrix[d].get(shop, 0)
                vals.append(c if c > 0 else "")
                tot += c
            vals.append(tot)
            self.summary_tree.insert("", "end", values=vals)

    def copy_to_clipboard(self):
        tab = self.notebook.index(self.notebook.select())
        tree = self.detail_tree if tab == 0 else self.summary_tree
        if tab == 0:
             cols = [self.detail_tree.heading(c)['text'] for c in self.detail_tree['columns']]
        else:
             cols = ["Date"] + [s.split("-")[0].title() for s in SHORT_SHOP_NAMES] + ["Total"]
        
        out = ["\t".join(cols)]
        for item in tree.get_children():
            out.append("\t".join(map(str, tree.item(item)['values'])))
            
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(out))
        messagebox.showinfo("Copied", "Copied to clipboard!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ReturnGoApp(root)
    root.mainloop()