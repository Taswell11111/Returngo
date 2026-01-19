import os
import customtkinter as ctk
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from datetime import datetime, timezone

# --- CONFIGURATION ---
MY_API_KEY = "G0dxSEprxa5iOAIsK5X6g7fCDdPl09q2882osmWD"

STORES = [
    {"name": "Diesel", "url": "diesel-dev-south-africa.myshopify.com"},
    {"name": "Hurley", "url": "hurley-dev-south-africa.myshopify.com"},
    {"name": "Jeep Apparel", "url": "jeep-apparel-dev-south-africa.myshopify.com"},
    {"name": "Reebok", "url": "reebok-dev-south-africa.myshopify.com"},
    {"name": "Superdry", "url": "superdry-dev-south-africa.myshopify.com"}
]

class RmaCentralFinal(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ReturnGO Multi-Store Operations 2026")
        self.geometry("1750x1000")
        
        # --- CONNECTION POOL ---
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        self._stop_event = threading.Event()
        self.searchable_rows = [] # Stores visible data for copying

        if not MY_API_KEY:
            self.after(500, lambda: self.log("❌ CRITICAL ERROR: 'ReturnGo_API' variable not found!", is_error=True))

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0)

        # --- TOP NAV ---
        self.nav = ctk.CTkFrame(self, fg_color="transparent")
        self.nav.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        
        self.lbl_search = ctk.CTkLabel(self.nav, text="🔍 Filter:", font=("Arial", 12, "bold"))
        self.lbl_search.pack(side="left", padx=(0, 5))
        self.search_entry = ctk.CTkEntry(self.nav, placeholder_text="Type RMA, Order #, or Tracking...", width=300)
        self.search_entry.pack(side="left")
        self.search_entry.bind("<KeyRelease>", self.filter_data)

        self.btn_refresh = ctk.CTkButton(self.nav, text="🔄 Sync All", command=self.start_sync, width=120)
        self.btn_refresh.pack(side="right", padx=5)
        self.btn_stop = ctk.CTkButton(self.nav, text="🛑 Stop", fg_color="#a12d2d", hover_color="#7a2222", command=self.stop_execution, width=100)
        self.btn_stop.pack(side="right", padx=5)

        # --- LAYER 1: STORE HEADERS ---
        self.header_container = ctk.CTkFrame(self, fg_color="transparent")
        self.header_container.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.store_widgets = {}
        
        for i, store in enumerate(STORES):
            self.header_container.grid_columnconfigure(i, weight=1)
            f = ctk.CTkFrame(self.header_container, border_width=2, border_color="#1f538d")
            f.grid(row=0, column=i, padx=5, sticky="nsew")
            
            ctk.CTkLabel(f, text=store["name"], font=("Arial", 14, "bold"), fg_color="#1f538d").pack(fill="x", pady=(0,5))
            
            stat_box = ctk.CTkFrame(f, fg_color="transparent")
            stat_box.pack(fill="both", expand=True, padx=5, pady=5)
            
            self.store_widgets[store["url"]] = {
                "Pending":  self.make_stat(stat_box, "Pending", store["url"], "Pending"),
                "Approved": self.make_stat(stat_box, "Approved", store["url"], "Approved"),
                "Received": self.make_stat(stat_box, "Received", store["url"], "Received"),
                "NoTrack":  self.make_stat(stat_box, "Approved (No Track)", store["url"], "NoTrack")
            }

        # --- LAYER 2: SINGLE TABLE ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=2, column=0, padx=20, pady=5, sticky="nsew")
        
        # Table Header Bar
        self.h_frame = ctk.CTkFrame(self.main_frame, fg_color="#1f538d", height=35)
        self.h_frame.pack(fill="x", padx=5, pady=(5,0))
        
        cols = [("#", 40), ("RMA ID", 120), ("Order #", 120), ("Tracking Number", 220), ("Created", 110), ("Updated", 110), ("Status", 100), ("Days Since", 80)]
        for text, width in cols:
            ctk.CTkLabel(self.h_frame, text=text, width=width, font=("Arial", 11, "bold"), anchor="w").pack(side="left", padx=5)
            ctk.CTkFrame(self.h_frame, width=2, height=15, fg_color="#888888").pack(side="left", padx=0)
            
        # Copy Button (Far Right of Header)
        self.btn_copy = ctk.CTkButton(self.h_frame, text="📋 Copy Table", width=100, height=24, fg_color="#2b2b2b", hover_color="#444", command=self.copy_table_data)
        self.btn_copy.pack(side="right", padx=10, pady=2)

        self.table_scroll = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        self.table_scroll.pack(fill="both", expand=True)

        # --- LAYER 3: LOGGER ---
        self.logger_text = ctk.CTkTextbox(self, height=150, font=("Consolas", 12), fg_color="#1a1a1a")
        self.logger_text.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
        self.logger_text.tag_config("error", foreground="#ff5555") 
        self.logger_text.tag_config("detail", foreground="#aaaaaa")
        
        self.log("System Initialized.")
        
        # --- AUTO SYNC START ---
        self.after(500, self.start_sync)

    def make_stat(self, parent, label, url, status):
        frame = ctk.CTkFrame(parent, border_width=1, border_color="#555555")
        frame.pack(fill="x", pady=3)
        btn = ctk.CTkButton(frame, text=f"{label}: --", fg_color="transparent", anchor="w", hover_color="#333333",
                            command=lambda: self.load_table(url, status))
        btn.pack(fill="both", padx=5, pady=2)
        return btn

    def log(self, message, is_error=False, is_detail=False):
        timestamp = datetime.now().strftime('%H:%M:%S')
        msg = f"[{timestamp}] {message}\n"
        tag = "error" if is_error else "detail" if is_detail else ""
        self.logger_text.insert("end", msg, tag)
        self.logger_text.see("end")

    def stop_execution(self):
        self._stop_event.set()
        self.log("🛑 STOP command received. Stopping worker threads...", is_error=True)
        self.btn_refresh.configure(state="normal")

    def start_sync(self):
        self._stop_event.clear()
        self.btn_refresh.configure(state="disabled")
        self.log("Starting full sync across all stores...")
        threading.Thread(target=self._sync_all_counts, daemon=True).start()

    def _sync_all_counts(self):
        for store in STORES:
            if self._stop_event.is_set(): break
            self.log(f"--- Syncing {store['name']} ---")
            headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": store["url"]}
            
            for status in ["Pending", "Approved", "Received"]:
                self.log(f"{store['name']}: Fetching {status}...", is_detail=True)
                try:
                    res = self.session.get(f"https://api.returngo.ai/rmas?status={status}&pagesize=50", headers=headers, timeout=10)
                    if res.status_code == 200:
                        data = res.json().get("rmas", [])
                        count = len(data)
                        display = f"{count}+" if count >= 50 else str(count)
                        self.store_widgets[store["url"]][status].configure(text=f"{status}: {display}")
                        self.log(f"{store['name']}: Found {count} {status}", is_detail=True)
                        
                        if status == "Approved":
                            no_track_count = 0
                            if count > 0:
                                self.log(f"{store['name']}: Checking tracking for {count} approved items...", is_detail=True)
                            
                            for rma in data:
                                if self._stop_event.is_set(): break
                                try:
                                    det = self.session.get(f"https://api.returngo.ai/rma/{rma['rmaId']}", headers=headers, timeout=10).json()
                                    shipments = det.get('shipments', [])
                                    if not shipments:
                                        no_track_count += 1
                                    elif all(not s.get('trackingNumber') for s in shipments):
                                        no_track_count += 1
                                except: pass
                            
                            self.store_widgets[store["url"]]["NoTrack"].configure(text=f"No Tracking: {no_track_count}")
                    else:
                        self.log(f"API Error {store['name']} {status}: {res.status_code}", is_error=True)
                except Exception as e:
                    self.log(f"Connection Error {store['name']}: {e}", is_error=True)
        
        self.log("Sync Complete. Ready.")
        self.btn_refresh.configure(state="normal")

    def load_table(self, url, status_type):
        self.log(f"Loading table for {status_type}...", is_error=False)
        for w in self.table_scroll.winfo_children(): w.destroy()
        self.searchable_rows = [] 
        threading.Thread(target=self._fetch_detailed_rmas, args=(url, status_type), daemon=True).start()

    def _fetch_detailed_rmas(self, url, status_type):
        headers = {"X-API-KEY": MY_API_KEY, "x-shop-name": url}
        api_status = "Approved" if status_type == "NoTrack" else status_type
        params = {"status": api_status, "pagesize": 50, "sort_by": "+rma_created_at"} 

        try:
            res = self.session.get(f"https://api.returngo.ai/rmas", headers=headers, params=params, timeout=10)
            if res.status_code == 200:
                rmas = res.json().get("rmas", [])
                self.log(f"Found {len(rmas)} records. Fetching details...")
                
                display_counter = 1
                for i, rma_sumry in enumerate(rmas):
                    if self._stop_event.is_set(): break
                    try:
                        det = self.session.get(f"https://api.returngo.ai/rma/{rma_sumry['rmaId']}", headers=headers, timeout=10).json()
                        
                        shipments = det.get('shipments', [])
                        track_nums = [s.get('trackingNumber') for s in shipments if s.get('trackingNumber')]
                        track_str = ", ".join(track_nums) if track_nums else "N/A"
                        
                        if status_type == "NoTrack" and track_nums:
                            continue 
                        
                        updated_str = det.get('lastUpdated')
                        days_since = "N/A"
                        if updated_str:
                            try:
                                dt_obj = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
                                delta = datetime.now(timezone.utc) - dt_obj
                                days_since = str(delta.days)
                            except: pass
                        
                        order_id = rma_sumry.get('order_name', 'N/A')
                        self.render_row(display_counter, rma_sumry, order_id, track_str, det, days_since)
                        display_counter += 1
                        
                    except Exception as e:
                        self.log(f"Error parsing row: {e}", is_error=True)
                
                self.log("Done.")
            else:
                self.log(f"API Error: {res.status_code}", is_error=True)
        except Exception as e:
            self.log(f"Critical Fetch Error: {e}", is_error=True)

    def render_row(self, idx, rma, order_id, track_str, det, days_since):
        row = ctk.CTkFrame(self.table_scroll)
        row.pack(fill="x", pady=2)
        
        def add_selectable_cell(text, w):
            e = ctk.CTkEntry(row, width=w, border_width=0, fg_color="transparent")
            e.insert(0, str(text))
            e.configure(state="readonly") 
            e.pack(side="left", padx=5)
            ctk.CTkFrame(row, width=2, height=20, fg_color="gray").pack(side="left", padx=0)

        def add_label_cell(text, w):
            ctk.CTkLabel(row, text=str(text), width=w, anchor="w").pack(side="left", padx=5)
            ctk.CTkFrame(row, width=2, height=20, fg_color="gray").pack(side="left", padx=0)

        add_label_cell(str(idx), 40)
        add_selectable_cell(rma.get('rmaId'), 120) 
        add_selectable_cell(order_id, 120)         
        add_selectable_cell(track_str, 220)        
        add_label_cell(det.get('createdAt', 'N/A')[:10], 110)
        add_label_cell(det.get('lastUpdated', 'N/A')[:10], 110)
        add_label_cell(rma.get('status'), 100)
        
        bg_color = "transparent"
        try:
            if int(days_since) > 7: bg_color = "#660000"
        except: pass
        days_lbl = ctk.CTkLabel(row, text=f"{days_since} days", width=80, fg_color=bg_color, corner_radius=5)
        days_lbl.pack(side="left", padx=5)

        ctk.CTkButton(row, text="View Timeline", width=110, height=24, 
                      command=lambda: self.show_timeline(det)).pack(side="right", padx=10)

        # Store data for Search AND Copy functionality
        row_data = {
            "widget": row, 
            "search_text": f"{rma.get('rmaId')} {order_id} {track_str}".lower(),
            "export_data": [str(idx), rma.get('rmaId'), order_id, track_str, det.get('createdAt', 'N/A')[:10], det.get('lastUpdated', 'N/A')[:10], rma.get('status'), days_since]
        }
        self.searchable_rows.append(row_data)

    def filter_data(self, event):
        query = self.search_entry.get().lower()
        for item in self.searchable_rows:
            if query in item["search_text"]:
                item["widget"].pack(fill="x", pady=2)
            else:
                item["widget"].pack_forget()

    def copy_table_data(self):
        # Header Row
        headers = ["#", "RMA ID", "Order #", "Tracking", "Created", "Updated", "Status", "Days Since"]
        tsv_output = ["\t".join(headers)]
        
        # Add filtered rows
        visible_rows = 0
        for item in self.searchable_rows:
            # Check if widget is visible (mapped)
            if item["widget"].winfo_ismapped():
                tsv_output.append("\t".join(item["export_data"]))
                visible_rows += 1
        
        # Copy to clipboard
        final_string = "\n".join(tsv_output)
        self.clipboard_clear()
        self.clipboard_append(final_string)
        self.log(f"Copied {visible_rows} rows to clipboard.", is_detail=True)

    def show_timeline(self, det):
        pop = ctk.CTkToplevel(self)
        pop.title(f"Timeline: {det.get('rmaSummary', {}).get('rmaId')}")
        pop.geometry("700x600")
        pop.attributes('-topmost', True)
        
        s = ctk.CTkScrollableFrame(pop, label_text="Timeline History")
        s.pack(fill="both", expand=True, padx=10, pady=10)
        
        for c in det.get('comments', []):
            f = ctk.CTkFrame(s, fg_color="#2b2b2b", corner_radius=6)
            f.pack(fill="x", pady=5, padx=(20, 5))
            
            ctk.CTkLabel(f, text=f"{c['triggeredBy']}  |  {c['datetime'][:16]}", 
                         font=("Arial", 11, "bold"), text_color="#aaaaaa", anchor="w").pack(fill="x", padx=10, pady=(5,0))
            
            ctk.CTkLabel(f, text=c['htmlText'], justify="left", wraplength=550, anchor="w").pack(fill="x", padx=10, pady=5)

if __name__ == "__main__":
    app = RmaCentralFinal()
    app.mainloop()