<<<<<<< HEAD
import textwrap
from collections import deque
import time
import customtkinter as ctk
from customtkinter import CTkFont
from tkinter import ttk
import traceback
from CTkMessagebox import CTkMessagebox
import os
import locale
import tkinter as tk
from tkinter import ttk, filedialog
import scapy.all as scapy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading
import time
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import networkx as nx
from scipy.stats import entropy
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.lib.units import inch
import io
from reportlab.lib.utils import ImageReader
from reportlab.graphics.shapes import Image as RLImage

# Set locale to 'C'
os.environ['LANG'] = 'C'
locale.setlocale(locale.LC_ALL, 'C')

class AdvancedAIS:
    def __init__(self):
        self.kmeans_model = None
        self.isolation_forest = None
        self.one_class_svm = None
        self.scaler = StandardScaler()
        self.anomaly_threshold = 0.3 # 95th percentile
        self.clonal_selection_memory = []
        self.negative_selection_detectors = []
        
    def train(self, data):
        if data.shape[0] == 0 or data.ndim != 2:
            raise ValueError(f"Invalid data shape: {data.shape}. Expected 2D array with at least one sample.")

        scaled_data = self.scaler.fit_transform(data)
        
        # K-Means clustering
        n_clusters = min(5, data.shape[0])
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans_model.fit(scaled_data)
        
        # Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(scaled_data)
        
        # One-Class SVM
        self.one_class_svm = OneClassSVM(kernel='rbf', nu=0.1)
        self.one_class_svm.fit(scaled_data)
        
        self.train_clonal_selection(scaled_data)
        self.generate_detectors(scaled_data)
        
    def train_clonal_selection(self, data, memory_size=100):
        for point in data:
            if len(self.clonal_selection_memory) < memory_size:
                self.clonal_selection_memory.append(point)
            elif np.random.random() < self.calculate_diversity(point):
                replace_index = np.random.randint(0, len(self.clonal_selection_memory))
                self.clonal_selection_memory[replace_index] = point
    
    def calculate_diversity(self, point):
        if not self.clonal_selection_memory:
            return 1.0
        distances = [np.linalg.norm(point - mem) for mem in self.clonal_selection_memory]
        return np.mean(distances) / np.max(distances)
    
    def generate_detectors(self, data, num_detectors=100):
        self.negative_selection_detectors = []
        for _ in range(num_detectors):
            detector = np.random.randn(data.shape[1])
            if self.is_valid_detector(detector, data):
                self.negative_selection_detectors.append(detector)
    
    def is_valid_detector(self, detector, data, threshold=0.95):
        distances = [np.linalg.norm(detector - point) for point in data]
        return np.min(distances) > threshold
        
    def detect_anomalies(self, new_data):
        if not self.is_trained():
            return np.array([False] * len(new_data)), np.array([0] * len(new_data))
        
        anomaly_scores = self.get_anomaly_score(new_data)
        is_anomaly = anomaly_scores > self.anomaly_threshold
        
        return is_anomaly, anomaly_scores
    
    def detect_anomaly_clonal_selection(self, point, threshold=0.9):
        if not self.clonal_selection_memory:
            return True
        distances = [np.linalg.norm(point - mem) for mem in self.clonal_selection_memory]
        return np.min(distances) > threshold
    
    def detect_anomaly_negative_selection(self, point, threshold=0.9):
        if not self.negative_selection_detectors:
            return False
        distances = [np.linalg.norm(point - detector) for detector in self.negative_selection_detectors]
        return np.min(distances) < threshold

    def get_anomaly_score(self, new_data):
        if not self.is_trained():
            return np.array([0] * len(new_data))
        
        scaled_new_data = self.scaler.transform(new_data)
        
        kmeans_scores = self.kmeans_model.transform(scaled_new_data).min(axis=1)
        if_scores = -self.isolation_forest.score_samples(scaled_new_data)
        svm_scores = -self.one_class_svm.score_samples(scaled_new_data)
        cs_scores = np.array([self.get_anomaly_score_clonal_selection(point) for point in scaled_new_data])
        ns_scores = np.array([self.get_anomaly_score_negative_selection(point) for point in scaled_new_data])
        
        # Normalize scores between 0 and 1
        kmeans_scores = (kmeans_scores - np.min(kmeans_scores)) / (np.max(kmeans_scores) - np.min(kmeans_scores) + 1e-10)
        if_scores = (if_scores - np.min(if_scores)) / (np.max(if_scores) - np.min(if_scores) + 1e-10)
        svm_scores = (svm_scores - np.min(svm_scores)) / (np.max(svm_scores) - np.min(svm_scores) + 1e-10)
        
        # Combine scores (you can adjust weights if needed)
        combined_scores = (kmeans_scores + if_scores + svm_scores + cs_scores + ns_scores) / 5
        
        return combined_scores
    
    def get_anomaly_score_clonal_selection(self, point):
        if not self.clonal_selection_memory:
            return 1.0
        distances = [np.linalg.norm(point - mem) for mem in self.clonal_selection_memory]
        return np.min(distances)
    
    def get_anomaly_score_negative_selection(self, point):
        if not self.negative_selection_detectors:
            return 0.0
        distances = [np.linalg.norm(point - detector) for detector in self.negative_selection_detectors]
        return 1 - np.min(distances)

    def is_trained(self):
        return self.kmeans_model is not None and self.isolation_forest is not None and self.one_class_svm is not None

class AdvancedNetworkAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Network Traffic Analyzer (AIS Integrated)")
        self.master.geometry("1600x1000")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.packet_buffer = deque(maxlen=1000)  # Buffer for batch updates
        self.last_update_time = time.time()
        self.update_interval = 1.0  # Update every 1 second
        
        self.create_widgets()
        self.initialize_data_structures()
        
        self.ais = AdvancedAIS()
        self.min_packets_for_training = 1
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.threat_levels = {"Low": 0, "Medium": 0, "High": 0}

    def initialize_data_structures(self):
        self.packets = []
        self.is_capturing = False
        self.protocol_counts = defaultdict(int)
        self.packet_sizes = []
        self.anomalies = []
        self.ip_connections = defaultdict(set)
        self.traffic_over_time = defaultdict(int)
        self.threat_levels = {"Low": 0, "Medium": 0, "High": 0}
        self.tcp_packets = []
        self.udp_packets = []
        self.other_packets = []

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.create_control_frame(main_frame)
        self.create_notebook(main_frame)

    def create_control_frame(self, parent):
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ctk.CTkButton(control_frame, text="Start Analysis", command=self.toggle_capture,font=CTkFont(size=14, weight="bold"))
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.report_button = ctk.CTkButton(control_frame, text="Generate Report", command=self.generate_report,state=tk.DISABLED, font=CTkFont(size=14, weight="bold"))
        self.report_button.pack(side=tk.LEFT, padx=10)

        self.train_ais_button = ctk.CTkButton(control_frame, text="Train AIS", command=self.train_ais,font=CTkFont(size=14, weight="bold"))
        self.train_ais_button.pack(side=tk.LEFT, padx=10)

        self.status_label = ctk.CTkLabel(control_frame, text="Status: Idle",font=CTkFont(size=14))
        self.status_label.pack(side=tk.RIGHT, padx=10)

        self.ais_status_label = ctk.CTkLabel(control_frame, text="AIS Status: Not Trained",font=CTkFont(size=14))
        self.ais_status_label.pack(side=tk.RIGHT, padx=10)

    def create_notebook(self, parent):
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background=self.master.cget('bg'))
        style.configure("TNotebook.Tab", padding=[10, 5], font=('Helvetica', 12))
        style.map("TNotebook.Tab", background=[("selected", "#3a7ebf")],foreground=[("selected", "#ffffff")])

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        self.create_packet_list_tab()
        self.create_graphs_tab()
        self.create_network_graph_tab()
        self.create_threat_dashboard_tab()
        self.create_ais_metrics_tab()
        self.create_traffic_over_time_tab()
        self.create_anomaly_chart_tab()
        self.create_ip_connections_tab()

    def create_packet_list_tab(self):
        packet_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(packet_frame, text="Packet List")

        style = ttk.Style()
        style.configure("Treeview", 
                        background="#2a2d2e", 
                        foreground="white", 
                        rowheight=25, 
                        fieldbackground="#343638", 
                        bordercolor="#343638", 
                        borderwidth=0)
        style.map('Treeview', background=[('selected', '#22559b')])
        style.configure("Treeview.Heading", 
                        background="#3a7ebf", 
                        foreground="white", 
                        relief="flat")
        style.map("Treeview.Heading",background=[('active', '#3a7ebf')])

        self.tree = ttk.Treeview(packet_frame, columns=("Time", "Source", "Destination", "Protocol", "Length", "Anomaly", "Anomaly Score", "Threat Level"), show="headings")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        tree_scrollbar = ttk.Scrollbar(packet_frame, orient="vertical", command=self.tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

    def create_graphs_tab(self):
        graphs_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(graphs_frame, text="Graphs")

        protocol_frame = ctk.CTkFrame(graphs_frame)
        protocol_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig_protocol, self.ax_protocol = plt.subplots(figsize=(6, 4))
        self.canvas_protocol = FigureCanvasTkAgg(self.fig_protocol, master=protocol_frame)
        self.canvas_protocol.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        size_frame = ctk.CTkFrame(graphs_frame)
        size_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig_size, self.ax_size = plt.subplots(figsize=(6, 4))
        self.canvas_size = FigureCanvasTkAgg(self.fig_size, master=size_frame)
        self.canvas_size.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_network_graph_tab(self):
        network_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(network_frame, text="Network Graph")

        self.network_graph = nx.Graph()
        self.network_pos = {}
        self.fig_network, self.ax_network = plt.subplots(figsize=(8, 6))
        self.canvas_network = FigureCanvasTkAgg(self.fig_network, master=network_frame)
        self.canvas_network.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_threat_dashboard_tab(self):
        threat_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(threat_frame, text="Threat Dashboard")

        self.fig_threat, self.ax_threat = plt.subplots(figsize=(8, 6))
        self.canvas_threat = FigureCanvasTkAgg(self.fig_threat, master=threat_frame)
        self.canvas_threat.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_ais_metrics_tab(self):
        ais_metrics_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(ais_metrics_frame, text="AIS Metrics")

        self.ais_metrics_text = ctk.CTkTextbox(ais_metrics_frame, height=400, width=600)
        self.ais_metrics_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def create_traffic_over_time_tab(self):
        traffic_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(traffic_frame, text="Traffic Over Time")

        self.fig_traffic, self.ax_traffic = plt.subplots(figsize=(8, 6))
        self.canvas_traffic = FigureCanvasTkAgg(self.fig_traffic,master=traffic_frame)
        self.canvas_traffic.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_anomaly_chart_tab(self):
        anomaly_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(anomaly_frame, text="Anomaly Chart")

        self.fig_anomaly, self.ax_anomaly = plt.subplots(figsize=(8, 6))
        self.canvas_anomaly = FigureCanvasTkAgg(self.fig_anomaly, master=anomaly_frame)
        self.canvas_anomaly.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_ip_connections_tab(self):
        ip_connections_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(ip_connections_frame, text="IP Connections")

        self.fig_ip_connections, self.ax_ip_connections = plt.subplots(figsize=(8, 6))
        self.canvas_ip_connections = FigureCanvasTkAgg(self.fig_ip_connections, master=ip_connections_frame)
        self.canvas_ip_connections.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def toggle_capture(self):
        if not self.is_capturing:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        self.is_capturing = True
        self.start_button.configure(text="Analizi Durdur")
        self.report_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Durum: Analiz Ediliyor")
        self.update_status("Paket yakalama başlatılıyor...")
        self.tree.delete(*self.tree.get_children())  # Listeyi temizle
        self.capture_thread = threading.Thread(target=self.capture_packets)
        self.capture_thread.start()
        self.update_gui()

    def stop_capture(self):
        self.is_capturing = False
        self.start_button.configure(text="Start Analysis")
        self.report_button.configure(state=tk.NORMAL)
        self.status_label.configure(text="Status: Analysis Stopped")
        self.update_status("Stopping packet capture...")

    def capture_packets(self):
        packet_count = 0
        self.update_status("Packet capture started.")
        while self.is_capturing:
            try:
                packets = scapy.sniff(count=1, timeout=1)
                self.update_status(f"{len(packets)} packets captured")
                for packet in packets:
                    if scapy.IP in packet:
                        self.process_packet(packet)
                        packet_count += 1
                        
                        if packet_count % 1 == 0:
                            self.update_status(f"{packet_count} packets captured and processed")
                        
                        if packet_count % 1 == 0:
                            self.train_ais()
            except Exception as e:
                self.update_status(f"Error capturing packets: {str(e)}")
                self.master.after(0, lambda: traceback.print_exc())
                time.sleep(1)

    def process_packet(self, packet):
        try:
            if scapy.IP in packet:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S")
                src = packet[scapy.IP].src
                dst = packet[scapy.IP].dst
                proto = packet[scapy.IP].proto
                length = len(packet)
                
                self.packets.append(packet)
                self.protocol_counts[proto] += 1
                self.packet_sizes.append(length)
                self.ip_connections[src].add(dst)
                
                current_minute = time.strftime("%Y-%m-%d %H:%M")
                self.traffic_over_time[current_minute] += length
                
                is_anomaly = False
                anomaly_score = 0.0
                threat_level = "Normal"
                
                if self.ais.is_trained():
                    features = [proto, length, self.calculate_entropy(packet)]
                    is_anomaly, anomaly_scores = self.ais.detect_anomalies([features])
                    is_anomaly = is_anomaly[0]
                    anomaly_score = anomaly_scores[0]
                    
                    threat_level = self.determine_threat_level(anomaly_score)
                    
                    if is_anomaly:
                        self.anomalies.append((time_str, src, dst, proto, length, f"Anomaly (Score: {anomaly_score:.2f})", threat_level))
                
                self.master.after(0, lambda: self.tree.insert("", 0, values=(time_str, src, dst, proto, length, "Yes" if is_anomaly else "No", f"{anomaly_score:.2f}", threat_level)))
                
                if scapy.TCP in packet:
                    self.tcp_packets.append((time_str, src, dst, length))
                elif scapy.UDP in packet:
                    self.udp_packets.append((time_str, src, dst, length))
                else:
                    self.other_packets.append((time_str, src, dst, proto, length))
                
                if len(self.packets) % self.min_packets_for_training == 0:
                    self.train_ais()
            
        except Exception as e:
            self.update_status(f"Error processing packet: {str(e)}")
            self.master.after(0, lambda: traceback.print_exc())
            
    def update_tree(self, time_str, src, dst, proto, length, is_anomaly, anomaly_score, threat_level):
        self.tree.insert("", 0, values=(time_str, src, dst, proto, length, "Yes" if is_anomaly else "No", f"{anomaly_score:.2f}", threat_level))
        

    def calculate_entropy(self, packet):
        if scapy.Raw in packet:
            payload = packet[scapy.Raw].load
            _, counts = np.unique(list(payload), return_counts=True)
            return entropy(counts, base=2)
        return 0

    def determine_threat_level(self, anomaly_score):
        if anomaly_score > 0.7:
            self.threat_levels["High"] += 1
            return "High"
        elif anomaly_score > 0.55:
            self.threat_levels["Medium"] += 1
            return "Medium"
        elif anomaly_score > 0.3:
            self.threat_levels["Low"] += 1
            return "Low"
        else:
            return "Normal"

    def update_gui(self):
        if self.is_capturing:
            self.update_protocol_chart()
            self.update_size_chart()
            self.update_network_graph()
            self.update_threat_dashboard()
            self.update_traffic_over_time()
            self.update_anomaly_chart()
            self.update_ip_connections_chart()
            self.master.update_idletasks()
            self.master.after(1000, self.update_gui)

    def update_packet_list(self):
        # Batch update the treeview
        for packet_info in self.packet_buffer:
            self.tree.insert("", 0, values=packet_info)
        self.packet_buffer.clear()
        
    def update_protocol_chart(self):
        self.ax_protocol.clear()
        if self.protocol_counts:
            protocols = list(self.protocol_counts.keys())
            counts = list(self.protocol_counts.values())
            protocol_names = [self.get_protocol_name(proto) for proto in protocols]
            wedges, texts, autotexts = self.ax_protocol.pie(counts, labels=protocols, autopct='%1.1f%%', startangle=90)
            self.ax_protocol.set_title("Protocol Distribution")
            
            labels = [f"{proto} ({name})" for proto, name in zip(protocols, protocol_names)]
            legend_labels = [f'{proto} ({count})' for proto, count in zip(protocols, counts)]
            
            self.ax_protocol.legend(wedges,legend_labels, title="Protocols", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            self.ax_protocol.legend(wedges, labels, title="Protocols", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        self.canvas_protocol.draw()

    def get_protocol_name(self, proto):
        # Map protocol numbers to names
        protocol_map = {
            1: "ICMP",
            6: "TCP",
            17: "UDP",
            2: "IGMP",
            8: "EGP",
            9: "IGP",
            47: "GRE",
            50: "ESP",
            51: "AH",
            88: "EIGRP",
            89: "OSPF",
            # Add more as needed
        }
        return protocol_map.get(proto, "Unknown")
    
    def update_size_chart(self):
        self.ax_size.clear()
        if self.packet_sizes:
            self.ax_size.hist(self.packet_sizes, bins=20, edgecolor='black')
            self.ax_size.set_title("Packet Size Distribution")
            self.ax_size.set_xlabel("Packet Size (bytes)")
            self.ax_size.set_ylabel("Frequency")
        self.canvas_size.draw()

    def update_network_graph(self):
        self.ax_network.clear()
        self.network_graph.clear()
        
        for src, destinations in self.ip_connections.items():
            self.network_graph.add_node(src)
            for dst in destinations:
                self.network_graph.add_edge(src, dst)
        
        self.network_pos = nx.spring_layout(self.network_graph)
        
        nx.draw(self.network_graph, self.network_pos, ax=self.ax_network, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        
        self.ax_network.set_title("Network Connections Graph")
        self.canvas_network.draw()

    def update_threat_dashboard(self):
        self.ax_threat.clear()
        
        threat_levels = list(self.threat_levels.keys())
        counts = list(self.threat_levels.values())
        colors = ['green', 'yellow', 'red']
        
        self.ax_threat.bar(threat_levels, counts, color=colors)
        self.ax_threat.set_title("Threat Level Distribution")
        self.ax_threat.set_xlabel("Threat Level")
        self.ax_threat.set_ylabel("Count")
        
        for i, v in enumerate(counts):
            self.ax_threat.text(i, v, str(v), ha='center', va='bottom')
        
        self.canvas_threat.draw()

    def update_traffic_over_time(self):
        self.ax_traffic.clear()
        
        times = list(self.traffic_over_time.keys())
        traffic = list(self.traffic_over_time.values())
        
        self.ax_traffic.plot(times, traffic, marker='o')
        self.ax_traffic.set_title("Traffic Over Time")
        self.ax_traffic.set_xlabel("Time")
        self.ax_traffic.set_ylabel("Traffic Volume (bytes)")
        self.ax_traffic.tick_params(axis='x', rotation=45)
        
        self.canvas_traffic.draw()

    def update_anomaly_chart(self):
        self.ax_anomaly.clear()
        if self.anomalies:
            anomaly_times = [a[0] for a in self.anomalies]
            anomaly_scores = [float(a[5].split(":")[1].strip()[:-1]) for a in self.anomalies]
            threat_levels = [a[6] for a in self.anomalies]
            
            # Farklı tehdit seviyeleri için renkler
            colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Normal': 'blue'}
            
            for time, score, level in zip(anomaly_times, anomaly_scores, threat_levels):
                self.ax_anomaly.scatter(time, score, c=colors[level], label=level, s=50, edgecolors='black')
            
            self.ax_anomaly.set_title("Anomaly Scores Over Time")
            self.ax_anomaly.set_xlabel("Time")
            self.ax_anomaly.set_ylabel("Anomaly Score")
            self.ax_anomaly.tick_params(axis='x', rotation=45)
            
            # Eşik değerini göster
            self.ax_anomaly.axhline(y=self.ais.anomaly_threshold, color='r', linestyle='--', label='Anomaly Threshold')
            
            # Izgara ekle
            self.ax_anomaly.grid(True, linestyle='--', alpha=0.7)
            
            # Tekrarlanan etiketleri kaldır ve göster
            handles, labels = self.ax_anomaly.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.ax_anomaly.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
            
            # Son 10 veri noktasını göster (eğer varsa)
            if len(anomaly_times) > 10:
                self.ax_anomaly.set_xlim(anomaly_times[-10], anomaly_times[-1])
                self.ax_anomaly.set_ylim(0, max(max(anomaly_scores[-10:]), self.ais.anomaly_threshold) * 1.1)
        else:
            self.ax_anomaly.text(0.5, 0.5, "No anomalies detected", ha='center', va='center')
        
        self.canvas_anomaly.draw()

    def update_ip_connections_chart(self):
        self.ax_ip_connections.clear()
        ip_counts = {ip: len(connections) for ip, connections in self.ip_connections.items()}
        ips = list(ip_counts.keys())
        counts = list(ip_counts.values())
        self.ax_ip_connections.bar(ips, counts)
        self.ax_ip_connections.set_title("IP Connection Counts")
        self.ax_ip_connections.set_xlabel("IP Address")
        self.ax_ip_connections.set_ylabel("Connection Count")
        self.ax_ip_connections.tick_params(axis='x', rotation=90)
        self.canvas_ip_connections.draw()

    def train_ais(self):
        if len(self.packets) < self.min_packets_for_training:
            self.update_status(f"Insufficient packets for training. Current: {len(self.packets)}, Required: {self.min_packets_for_training}")
            self.ais_status_label.configure(text=f"AIS Status: Insufficient data ({len(self.packets)}/{self.min_packets_for_training})")
            return

        data = self.get_training_data()
        self.update_status(f"Training data shape: {data.shape}")
        
        if data.shape[0] == 0:
            self.update_status("No valid data for training. Collecting more packets...")
            self.ais_status_label.configure(text="AIS Status: No valid data")
            return
        
        try:
            self.update_status("Advanced AIS training started.")
            self.ais.train(data)
            self.update_status("Advanced AIS training completed.")
            self.ais_status_label.configure(text="AIS Status: Trained")
            self.show_ais_metrics()
        except ValueError as ve:
            self.update_status(f"ValueError during AIS training: {str(ve)}")
            self.ais_status_label.configure(text="AIS Status: Training error (ValueError)")
            self.master.after(0, lambda: traceback.print_exc())
        except Exception as e:
            self.update_status(f"Unexpected error during AIS training: {str(e)}")
            self.ais_status_label.configure(text="AIS Status: Training error (Unexpected)")
            self.master.after(0, lambda: traceback.print_exc())

    def get_training_data(self):
        self.update_status(f"Total captured packets: {len(self.packets)}")
        data = []
        for i, packet in enumerate(self.packets):
            if scapy.IP in packet:
                proto = packet[scapy.IP].proto
                length = len(packet)
                payload_entropy = self.calculate_entropy(packet)
                data.append([proto, length, payload_entropy])
                if i % 1 == 0 and i > 0:
                    self.update_status(f"Processed packet: {i}")
        
        self.update_status(f"Total processed packets: {len(data)}")
        
        if not data:
            self.update_status("No data found for training.")
            return np.array([]).reshape(0, 3)  # Empty 2D array with 3 features
        
        return np.array(data)

    def show_ais_metrics(self):
        if not self.ais.is_trained():
            self.ais_metrics_text.delete("1.0", tk.END)
            self.ais_metrics_text.insert(tk.END, "AIS Metrics: Not yet trained")
            return

        metrics_text = f"""AIS Metrics:
        K-Means Clustering:
        - Number of Clusters: {self.ais.kmeans_model.n_clusters}
        - Inertia: {self.ais.kmeans_model.inertia_:.2f}
        - Iterations: {self.ais.kmeans_model.n_iter_}

        Isolation Forest:
        - Number of Estimators: {self.ais.isolation_forest.n_estimators}
        - Contamination: {self.ais.isolation_forest.contamination}

        One-Class SVM:
        - Kernel: {self.ais.one_class_svm.kernel}
        - Nu: {self.ais.one_class_svm.nu}

        Clonal Selection:
        - Memory Size: {len(self.ais.clonal_selection_memory)}

        Negative Selection:
        - Number of Detectors: {len(self.ais.negative_selection_detectors)}

        Feature Scaling (StandardScaler):
        - Mean: {self.ais.scaler.mean_}
        - Variance: {self.ais.scaler.var_}
        """
        
        self.ais_metrics_text.delete("1.0", tk.END)
        self.ais_metrics_text.insert(tk.END, metrics_text)

    def generate_report(self):
        try:
            self.update_status("Starting report generation process...")
            
            pdfmetrics.registerFont(TTFont('Arial', 'C:\\Windows\\Fonts\\arial.ttf'))
            pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:\\Windows\\Fonts\\arialbd.ttf'))
            
            filename = filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF files", "*.pdf")])
            if not filename:
                self.update_status("File selection cancelled")
                
                return
            self.update_status(f"Selected file: {filename}")

            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='CustomHeading1',
                                    parent=styles['Heading1'],
                                    fontSize=14,
                                    spaceAfter=10))

            elements = []

            elements.append(Paragraph("Network Traffic Analysis Report", styles['Heading1']))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("1. Overview", styles['Heading2']))
            elements.append(Paragraph(f"Total Packets Analyzed: {len(self.packets)}", styles['BodyText']))
            elements.append(Paragraph(f"Analysis Duration: {self.get_analysis_duration()}", styles['BodyText']))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("2. Protocol Distribution", styles['Heading2']))
            elements.append(self.create_protocol_chart())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("3. Packet Size Distribution", styles['Heading2']))
            elements.append(self.create_packet_size_chart())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("4. Top IP Connections", styles['Heading2']))
            elements.append(self.create_ip_connections_table())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("5. Traffic Over Time", styles['Heading2']))
            elements.append(self.create_traffic_over_time_chart())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("6. Anomaly Detection", styles['Heading2']))
            elements.append(Paragraph(f"Total Anomalies Detected: {len(self.anomalies)}", styles['BodyText']))
            elements.append(self.create_anomalies_table())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("7. Threat Level Distribution", styles['Heading2']))
            elements.append(self.create_threat_level_chart())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("8. AIS Metrics", styles['Heading2']))
            elements.append(self.get_ais_metrics_table())
            elements.append(Spacer(1, 12))
            
            elements.append(Paragraph("9. Network Connections Graph", styles['Heading2']))
            network_graph_image = self.create_network_graph_for_report()
            elements.append(network_graph_image)
            elements.append(Spacer(1, 12))
            elements.append(self.create_network_analysis_table())
            elements.append(Spacer(1, 12))

            doc.build(elements)
            self.update_status(f"Report generated successfully: {filename}")
            CTkMessagebox(title="Success", message=f"Report generated successfully: {filename}", icon="check")
            
        except Exception as e:
            self.update_status(f"Error generating report: {str(e)}")
            self.master.after(0, lambda: traceback.print_exc())
            CTkMessagebox(title="Error", message=f"Error generating report: {str(e)}", icon="cancel")

    def get_analysis_duration(self):
        if not self.packets:
            return "No packets captured"
        start_time = self.packets[0].time
        end_time = self.packets[-1].time
        duration = end_time - start_time
        return f"{duration:.2f} seconds"

    def create_protocol_chart(self):
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 50
        pie.width = 100
        pie.height = 100
        pie.data = list(self.protocol_counts.values())
        pie.labels = [str(key) for key in self.protocol_counts.keys()]  # Sayısal değerleri stringe çevirin
        drawing.add(pie)
        return drawing

    def create_packet_size_chart(self):
        drawing = Drawing(400, 200)
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 300
        bc.data = [self.packet_sizes]
        bc.categoryAxis.categoryNames = [str(i) for i in range(len(self.packet_sizes))]
        drawing.add(bc)
        return drawing

    def create_ip_connections_table(self):
        data = [["Source IP", "Destination IP", "Connection Count"]]
        for src, destinations in self.ip_connections.items():
            for dst in destinations:
                data.append([src, dst, len(destinations)])
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Arial-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Arial'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table

    def create_traffic_over_time_chart(self):
        drawing = Drawing(400, 200)
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 300
        
        # Sadece saat bilgisini al
        times = [time.split()[1] for time in self.traffic_over_time.keys()]
        traffic = list(self.traffic_over_time.values())
        
        bc.data = [traffic]
        bc.categoryAxis.categoryNames = times
        bc.categoryAxis.labels.boxAnchor = 'ne'
        bc.categoryAxis.labels.angle = 45
        bc.categoryAxis.labels.dx = -8
        bc.categoryAxis.labels.dy = -2
        
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max(traffic) * 1.1
        bc.valueAxis.valueStep = max(traffic) // 5
        
        bc.bars[0].fillColor = colors.lightblue
        
        drawing.add(bc)
        
        # Başlık ekle
        title = String(200, 180, 'Traffic Over Time', fontSize=16, textAnchor='middle')
        drawing.add(title)
        
        return drawing

    def create_anomalies_table(self):
        data = [["Time", "Source IP", "Destination IP", "Protocol", "Length", "Status", "Threat Level","Explanation"]]
        for a in self.anomalies[:10]:  # Sadece ilk 10 anomaliyi göster
            time, src, dst, proto, length, status, threat_level = a[:7]
            explanation = self.get_anomaly_explanation(a)
            data.append([time, src, dst, proto, length, status, threat_level, explanation])
        
        table = Table(data, colWidths=[90, 70, 75, 40, 40, 85, 50,155])
        
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Arial-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Arial'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]
        
        # Tehdit seviyesine göre renklendirme
        for i, row in enumerate(data[1:], start=1):
            if row[6] == "High":
                style.append(('BACKGROUND', (6, i), (6, i), colors.red))
            elif row[6] == "Medium":
                style.append(('BACKGROUND', (6, i), (6, i), colors.orange))
            elif row[6] == "Low":
                style.append(('BACKGROUND', (6, i), (6, i), colors.yellow))
        
        table.setStyle(TableStyle(style))
        return table

    def get_anomaly_explanation(self, anomaly):
        _, _, _, proto, length, status, threat_level = anomaly[:7]
        anomaly_score = float(status.split(":")[1].strip()[:-1])
        
        explanations = []
        
        if anomaly_score > 0.8:
            explanations.append("Çok yüksek anomali skoru")
        elif anomaly_score > 0.6:
            explanations.append("Yüksek anomali skoru")
        
        if length > 1500:
            explanations.append("Olağandışı büyük paket boyutu")
        elif length < 20:
            explanations.append("Olağandışı küçük paket boyutu")
        
        if proto not in [6, 17]:  # TCP veya UDP değilse
            explanations.append("Nadir görülen protokol")
        
        if not explanations:
            explanations.append("Genel anormallik tespit edildi")
        
        return ", ".join(explanations)

    def create_threat_level_chart(self):
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 50
        pie.width = 100
        pie.height = 100
        pie.data = list(self.threat_levels.values())
        pie.labels = list(self.threat_levels.keys())
        drawing.add(pie)
        return drawing


    def get_ais_metrics_table(self):
        if not self.ais.is_trained():
            return Paragraph("AIS Metrics: Not yet trained", getSampleStyleSheet()['BodyText'])

        data = [
            ["AIS Metrics", ""],
            ["K-Means Clustering", ""],
            ["Number of Clusters", str(self.ais.kmeans_model.n_clusters)],
            ["Inertia", f"{self.ais.kmeans_model.inertia_:.2f}"],
            ["Iterations", str(self.ais.kmeans_model.n_iter_)],
            ["Isolation Forest", ""],
            ["Number of Estimators", str(self.ais.isolation_forest.n_estimators)],
            ["Contamination", str(self.ais.isolation_forest.contamination)],
            ["One-Class SVM", ""],
            ["Kernel", self.ais.one_class_svm.kernel],
            ["Nu", str(self.ais.one_class_svm.nu)],
            ["Clonal Selection", ""],
            ["Memory Size", str(len(self.ais.clonal_selection_memory))],
            ["Negative Selection", ""],
            ["Number of Detectors", str(len(self.ais.negative_selection_detectors))],
            ["Feature Scaling (StandardScaler)", ""],
            ["Mean", str(self.ais.scaler.mean_)],
            ["Variance", str(self.ais.scaler.var_)]
        ]

        table = Table(data, colWidths=[200, 300])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (0, 1), colors.lightgrey),
            ('BACKGROUND', (0, 5), (0, 5), colors.lightgrey),
            ('BACKGROUND', (0, 8), (0, 8), colors.lightgrey),
            ('BACKGROUND', (0, 11), (0, 11), colors.lightgrey),
            ('BACKGROUND', (0, 13), (0, 13), colors.lightgrey),
            ('BACKGROUND', (0, 15), (0, 15), colors.lightgrey),
        ]))

        return table

    def update_status(self, message):
        self.master.after(0, lambda: self.status_label.configure(text=f"Status: {message}"))

    def create_network_graph_for_report(self):
        plt.figure(figsize=(8, 6))
        nx.draw(self.network_graph, self.network_pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        plt.title("Network Connections Graph")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        img = Image(buf)
        img.drawHeight = 300
        img.drawWidth = 400
        
        return img

    def create_network_analysis_table(self):
        num_nodes = self.network_graph.number_of_nodes()
        num_edges = self.network_graph.number_of_edges()
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        data = [
            ["Network Analysis Metrics", "Value"],
            ["Total Number of Nodes", str(num_nodes)],
            ["Total Number of Edges", str(num_edges)],
            ["Average Degree", f"{avg_degree:.2f}"],
            ["Network Density", f"{nx.density(self.network_graph):.4f}"],
            ["Number of Connected Components", str(nx.number_connected_components(self.network_graph))],
        ]
        
        # Find nodes with highest degree
        degree_dict = dict(self.network_graph.degree())
        top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:5]
        
        data.append(["Nodes with Highest Degree", ""])
        for node in top_nodes:
            data.append(["", f"{node} ({degree_dict[node]})"])

        table = Table(data, colWidths=[250, 250])
        table.setStyle(TableStyle([
            # Header style
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            # Body style
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            # Lines
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
            ('LINEAFTER', (0, 0), (0, -1), 1, colors.black),
            # Specific styles
            ('BACKGROUND', (0, -6), (0, -1), colors.lightgrey),
            ('SPAN', (0, -6), (0, -1)),
        ]))
        
        return table
    
    def on_closing(self):
        self.is_capturing = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join()
        self.master.quit()
        self.master.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = AdvancedNetworkAnalyzer(root)
    root.mainloop()
=======
import textwrap
from collections import deque
import time
import customtkinter as ctk
from customtkinter import CTkFont
from tkinter import ttk
import traceback
from CTkMessagebox import CTkMessagebox
import os
import locale
import tkinter as tk
from tkinter import ttk, filedialog
import scapy.all as scapy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading
import time
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import networkx as nx
from scipy.stats import entropy
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.lib.units import inch
import io
from reportlab.lib.utils import ImageReader
from reportlab.graphics.shapes import Image as RLImage

# Set locale to 'C'
os.environ['LANG'] = 'C'
locale.setlocale(locale.LC_ALL, 'C')

class AdvancedAIS:
    def __init__(self):
        self.kmeans_model = None
        self.isolation_forest = None
        self.one_class_svm = None
        self.scaler = StandardScaler()
        self.anomaly_threshold = 0.3 # 95th percentile
        self.clonal_selection_memory = []
        self.negative_selection_detectors = []
        
    def train(self, data):
        if data.shape[0] == 0 or data.ndim != 2:
            raise ValueError(f"Invalid data shape: {data.shape}. Expected 2D array with at least one sample.")

        scaled_data = self.scaler.fit_transform(data)
        
        # K-Means clustering
        n_clusters = min(5, data.shape[0])
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans_model.fit(scaled_data)
        
        # Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(scaled_data)
        
        # One-Class SVM
        self.one_class_svm = OneClassSVM(kernel='rbf', nu=0.1)
        self.one_class_svm.fit(scaled_data)
        
        self.train_clonal_selection(scaled_data)
        self.generate_detectors(scaled_data)
        
    def train_clonal_selection(self, data, memory_size=100):
        for point in data:
            if len(self.clonal_selection_memory) < memory_size:
                self.clonal_selection_memory.append(point)
            elif np.random.random() < self.calculate_diversity(point):
                replace_index = np.random.randint(0, len(self.clonal_selection_memory))
                self.clonal_selection_memory[replace_index] = point
    
    def calculate_diversity(self, point):
        if not self.clonal_selection_memory:
            return 1.0
        distances = [np.linalg.norm(point - mem) for mem in self.clonal_selection_memory]
        return np.mean(distances) / np.max(distances)
    
    def generate_detectors(self, data, num_detectors=100):
        self.negative_selection_detectors = []
        for _ in range(num_detectors):
            detector = np.random.randn(data.shape[1])
            if self.is_valid_detector(detector, data):
                self.negative_selection_detectors.append(detector)
    
    def is_valid_detector(self, detector, data, threshold=0.95):
        distances = [np.linalg.norm(detector - point) for point in data]
        return np.min(distances) > threshold
        
    def detect_anomalies(self, new_data):
        if not self.is_trained():
            return np.array([False] * len(new_data)), np.array([0] * len(new_data))
        
        anomaly_scores = self.get_anomaly_score(new_data)
        is_anomaly = anomaly_scores > self.anomaly_threshold
        
        return is_anomaly, anomaly_scores
    
    def detect_anomaly_clonal_selection(self, point, threshold=0.9):
        if not self.clonal_selection_memory:
            return True
        distances = [np.linalg.norm(point - mem) for mem in self.clonal_selection_memory]
        return np.min(distances) > threshold
    
    def detect_anomaly_negative_selection(self, point, threshold=0.9):
        if not self.negative_selection_detectors:
            return False
        distances = [np.linalg.norm(point - detector) for detector in self.negative_selection_detectors]
        return np.min(distances) < threshold

    def get_anomaly_score(self, new_data):
        if not self.is_trained():
            return np.array([0] * len(new_data))
        
        scaled_new_data = self.scaler.transform(new_data)
        
        kmeans_scores = self.kmeans_model.transform(scaled_new_data).min(axis=1)
        if_scores = -self.isolation_forest.score_samples(scaled_new_data)
        svm_scores = -self.one_class_svm.score_samples(scaled_new_data)
        cs_scores = np.array([self.get_anomaly_score_clonal_selection(point) for point in scaled_new_data])
        ns_scores = np.array([self.get_anomaly_score_negative_selection(point) for point in scaled_new_data])
        
        # Normalize scores between 0 and 1
        kmeans_scores = (kmeans_scores - np.min(kmeans_scores)) / (np.max(kmeans_scores) - np.min(kmeans_scores) + 1e-10)
        if_scores = (if_scores - np.min(if_scores)) / (np.max(if_scores) - np.min(if_scores) + 1e-10)
        svm_scores = (svm_scores - np.min(svm_scores)) / (np.max(svm_scores) - np.min(svm_scores) + 1e-10)
        
        # Combine scores (you can adjust weights if needed)
        combined_scores = (kmeans_scores + if_scores + svm_scores + cs_scores + ns_scores) / 5
        
        return combined_scores
    
    def get_anomaly_score_clonal_selection(self, point):
        if not self.clonal_selection_memory:
            return 1.0
        distances = [np.linalg.norm(point - mem) for mem in self.clonal_selection_memory]
        return np.min(distances)
    
    def get_anomaly_score_negative_selection(self, point):
        if not self.negative_selection_detectors:
            return 0.0
        distances = [np.linalg.norm(point - detector) for detector in self.negative_selection_detectors]
        return 1 - np.min(distances)

    def is_trained(self):
        return self.kmeans_model is not None and self.isolation_forest is not None and self.one_class_svm is not None

class AdvancedNetworkAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Network Traffic Analyzer (AIS Integrated)")
        self.master.geometry("1600x1000")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.packet_buffer = deque(maxlen=1000)  # Buffer for batch updates
        self.last_update_time = time.time()
        self.update_interval = 1.0  # Update every 1 second
        
        self.create_widgets()
        self.initialize_data_structures()
        
        self.ais = AdvancedAIS()
        self.min_packets_for_training = 1
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.threat_levels = {"Low": 0, "Medium": 0, "High": 0}

    def initialize_data_structures(self):
        self.packets = []
        self.is_capturing = False
        self.protocol_counts = defaultdict(int)
        self.packet_sizes = []
        self.anomalies = []
        self.ip_connections = defaultdict(set)
        self.traffic_over_time = defaultdict(int)
        self.threat_levels = {"Low": 0, "Medium": 0, "High": 0}
        self.tcp_packets = []
        self.udp_packets = []
        self.other_packets = []

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.create_control_frame(main_frame)
        self.create_notebook(main_frame)

    def create_control_frame(self, parent):
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ctk.CTkButton(control_frame, text="Start Analysis", command=self.toggle_capture,font=CTkFont(size=14, weight="bold"))
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.report_button = ctk.CTkButton(control_frame, text="Generate Report", command=self.generate_report,state=tk.DISABLED, font=CTkFont(size=14, weight="bold"))
        self.report_button.pack(side=tk.LEFT, padx=10)

        self.train_ais_button = ctk.CTkButton(control_frame, text="Train AIS", command=self.train_ais,font=CTkFont(size=14, weight="bold"))
        self.train_ais_button.pack(side=tk.LEFT, padx=10)

        self.status_label = ctk.CTkLabel(control_frame, text="Status: Idle",font=CTkFont(size=14))
        self.status_label.pack(side=tk.RIGHT, padx=10)

        self.ais_status_label = ctk.CTkLabel(control_frame, text="AIS Status: Not Trained",font=CTkFont(size=14))
        self.ais_status_label.pack(side=tk.RIGHT, padx=10)

    def create_notebook(self, parent):
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background=self.master.cget('bg'))
        style.configure("TNotebook.Tab", padding=[10, 5], font=('Helvetica', 12))
        style.map("TNotebook.Tab", background=[("selected", "#3a7ebf")],foreground=[("selected", "#ffffff")])

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        self.create_packet_list_tab()
        self.create_graphs_tab()
        self.create_network_graph_tab()
        self.create_threat_dashboard_tab()
        self.create_ais_metrics_tab()
        self.create_traffic_over_time_tab()
        self.create_anomaly_chart_tab()
        self.create_ip_connections_tab()

    def create_packet_list_tab(self):
        packet_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(packet_frame, text="Packet List")

        style = ttk.Style()
        style.configure("Treeview", 
                        background="#2a2d2e", 
                        foreground="white", 
                        rowheight=25, 
                        fieldbackground="#343638", 
                        bordercolor="#343638", 
                        borderwidth=0)
        style.map('Treeview', background=[('selected', '#22559b')])
        style.configure("Treeview.Heading", 
                        background="#3a7ebf", 
                        foreground="white", 
                        relief="flat")
        style.map("Treeview.Heading",background=[('active', '#3a7ebf')])

        self.tree = ttk.Treeview(packet_frame, columns=("Time", "Source", "Destination", "Protocol", "Length", "Anomaly", "Anomaly Score", "Threat Level"), show="headings")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        tree_scrollbar = ttk.Scrollbar(packet_frame, orient="vertical", command=self.tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

    def create_graphs_tab(self):
        graphs_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(graphs_frame, text="Graphs")

        protocol_frame = ctk.CTkFrame(graphs_frame)
        protocol_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig_protocol, self.ax_protocol = plt.subplots(figsize=(6, 4))
        self.canvas_protocol = FigureCanvasTkAgg(self.fig_protocol, master=protocol_frame)
        self.canvas_protocol.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        size_frame = ctk.CTkFrame(graphs_frame)
        size_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig_size, self.ax_size = plt.subplots(figsize=(6, 4))
        self.canvas_size = FigureCanvasTkAgg(self.fig_size, master=size_frame)
        self.canvas_size.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_network_graph_tab(self):
        network_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(network_frame, text="Network Graph")

        self.network_graph = nx.Graph()
        self.network_pos = {}
        self.fig_network, self.ax_network = plt.subplots(figsize=(8, 6))
        self.canvas_network = FigureCanvasTkAgg(self.fig_network, master=network_frame)
        self.canvas_network.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_threat_dashboard_tab(self):
        threat_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(threat_frame, text="Threat Dashboard")

        self.fig_threat, self.ax_threat = plt.subplots(figsize=(8, 6))
        self.canvas_threat = FigureCanvasTkAgg(self.fig_threat, master=threat_frame)
        self.canvas_threat.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_ais_metrics_tab(self):
        ais_metrics_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(ais_metrics_frame, text="AIS Metrics")

        self.ais_metrics_text = ctk.CTkTextbox(ais_metrics_frame, height=400, width=600)
        self.ais_metrics_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def create_traffic_over_time_tab(self):
        traffic_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(traffic_frame, text="Traffic Over Time")

        self.fig_traffic, self.ax_traffic = plt.subplots(figsize=(8, 6))
        self.canvas_traffic = FigureCanvasTkAgg(self.fig_traffic,master=traffic_frame)
        self.canvas_traffic.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_anomaly_chart_tab(self):
        anomaly_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(anomaly_frame, text="Anomaly Chart")

        self.fig_anomaly, self.ax_anomaly = plt.subplots(figsize=(8, 6))
        self.canvas_anomaly = FigureCanvasTkAgg(self.fig_anomaly, master=anomaly_frame)
        self.canvas_anomaly.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_ip_connections_tab(self):
        ip_connections_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(ip_connections_frame, text="IP Connections")

        self.fig_ip_connections, self.ax_ip_connections = plt.subplots(figsize=(8, 6))
        self.canvas_ip_connections = FigureCanvasTkAgg(self.fig_ip_connections, master=ip_connections_frame)
        self.canvas_ip_connections.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def toggle_capture(self):
        if not self.is_capturing:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        self.is_capturing = True
        self.start_button.configure(text="Analizi Durdur")
        self.report_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Durum: Analiz Ediliyor")
        self.update_status("Paket yakalama başlatılıyor...")
        self.tree.delete(*self.tree.get_children())  # Listeyi temizle
        self.capture_thread = threading.Thread(target=self.capture_packets)
        self.capture_thread.start()
        self.update_gui()

    def stop_capture(self):
        self.is_capturing = False
        self.start_button.configure(text="Start Analysis")
        self.report_button.configure(state=tk.NORMAL)
        self.status_label.configure(text="Status: Analysis Stopped")
        self.update_status("Stopping packet capture...")

    def capture_packets(self):
        packet_count = 0
        self.update_status("Packet capture started.")
        while self.is_capturing:
            try:
                packets = scapy.sniff(count=1, timeout=1)
                self.update_status(f"{len(packets)} packets captured")
                for packet in packets:
                    if scapy.IP in packet:
                        self.process_packet(packet)
                        packet_count += 1
                        
                        if packet_count % 1 == 0:
                            self.update_status(f"{packet_count} packets captured and processed")
                        
                        if packet_count % 1 == 0:
                            self.train_ais()
            except Exception as e:
                self.update_status(f"Error capturing packets: {str(e)}")
                self.master.after(0, lambda: traceback.print_exc())
                time.sleep(1)

    def process_packet(self, packet):
        try:
            if scapy.IP in packet:
                time_str = time.strftime("%Y-%m-%d %H:%M:%S")
                src = packet[scapy.IP].src
                dst = packet[scapy.IP].dst
                proto = packet[scapy.IP].proto
                length = len(packet)
                
                self.packets.append(packet)
                self.protocol_counts[proto] += 1
                self.packet_sizes.append(length)
                self.ip_connections[src].add(dst)
                
                current_minute = time.strftime("%Y-%m-%d %H:%M")
                self.traffic_over_time[current_minute] += length
                
                is_anomaly = False
                anomaly_score = 0.0
                threat_level = "Normal"
                
                if self.ais.is_trained():
                    features = [proto, length, self.calculate_entropy(packet)]
                    is_anomaly, anomaly_scores = self.ais.detect_anomalies([features])
                    is_anomaly = is_anomaly[0]
                    anomaly_score = anomaly_scores[0]
                    
                    threat_level = self.determine_threat_level(anomaly_score)
                    
                    if is_anomaly:
                        self.anomalies.append((time_str, src, dst, proto, length, f"Anomaly (Score: {anomaly_score:.2f})", threat_level))
                
                self.master.after(0, lambda: self.tree.insert("", 0, values=(time_str, src, dst, proto, length, "Yes" if is_anomaly else "No", f"{anomaly_score:.2f}", threat_level)))
                
                if scapy.TCP in packet:
                    self.tcp_packets.append((time_str, src, dst, length))
                elif scapy.UDP in packet:
                    self.udp_packets.append((time_str, src, dst, length))
                else:
                    self.other_packets.append((time_str, src, dst, proto, length))
                
                if len(self.packets) % self.min_packets_for_training == 0:
                    self.train_ais()
            
        except Exception as e:
            self.update_status(f"Error processing packet: {str(e)}")
            self.master.after(0, lambda: traceback.print_exc())
            
    def update_tree(self, time_str, src, dst, proto, length, is_anomaly, anomaly_score, threat_level):
        self.tree.insert("", 0, values=(time_str, src, dst, proto, length, "Yes" if is_anomaly else "No", f"{anomaly_score:.2f}", threat_level))
        

    def calculate_entropy(self, packet):
        if scapy.Raw in packet:
            payload = packet[scapy.Raw].load
            _, counts = np.unique(list(payload), return_counts=True)
            return entropy(counts, base=2)
        return 0

    def determine_threat_level(self, anomaly_score):
        if anomaly_score > 0.7:
            self.threat_levels["High"] += 1
            return "High"
        elif anomaly_score > 0.55:
            self.threat_levels["Medium"] += 1
            return "Medium"
        elif anomaly_score > 0.3:
            self.threat_levels["Low"] += 1
            return "Low"
        else:
            return "Normal"

    def update_gui(self):
        if self.is_capturing:
            self.update_protocol_chart()
            self.update_size_chart()
            self.update_network_graph()
            self.update_threat_dashboard()
            self.update_traffic_over_time()
            self.update_anomaly_chart()
            self.update_ip_connections_chart()
            self.master.update_idletasks()
            self.master.after(1000, self.update_gui)

    def update_packet_list(self):
        # Batch update the treeview
        for packet_info in self.packet_buffer:
            self.tree.insert("", 0, values=packet_info)
        self.packet_buffer.clear()
        
    def update_protocol_chart(self):
        self.ax_protocol.clear()
        if self.protocol_counts:
            protocols = list(self.protocol_counts.keys())
            counts = list(self.protocol_counts.values())
            protocol_names = [self.get_protocol_name(proto) for proto in protocols]
            wedges, texts, autotexts = self.ax_protocol.pie(counts, labels=protocols, autopct='%1.1f%%', startangle=90)
            self.ax_protocol.set_title("Protocol Distribution")
            
            labels = [f"{proto} ({name})" for proto, name in zip(protocols, protocol_names)]
            legend_labels = [f'{proto} ({count})' for proto, count in zip(protocols, counts)]
            
            self.ax_protocol.legend(wedges,legend_labels, title="Protocols", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            self.ax_protocol.legend(wedges, labels, title="Protocols", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        self.canvas_protocol.draw()

    def get_protocol_name(self, proto):
        # Map protocol numbers to names
        protocol_map = {
            1: "ICMP",
            6: "TCP",
            17: "UDP",
            2: "IGMP",
            8: "EGP",
            9: "IGP",
            47: "GRE",
            50: "ESP",
            51: "AH",
            88: "EIGRP",
            89: "OSPF",
            # Add more as needed
        }
        return protocol_map.get(proto, "Unknown")
    
    def update_size_chart(self):
        self.ax_size.clear()
        if self.packet_sizes:
            self.ax_size.hist(self.packet_sizes, bins=20, edgecolor='black')
            self.ax_size.set_title("Packet Size Distribution")
            self.ax_size.set_xlabel("Packet Size (bytes)")
            self.ax_size.set_ylabel("Frequency")
        self.canvas_size.draw()

    def update_network_graph(self):
        self.ax_network.clear()
        self.network_graph.clear()
        
        for src, destinations in self.ip_connections.items():
            self.network_graph.add_node(src)
            for dst in destinations:
                self.network_graph.add_edge(src, dst)
        
        self.network_pos = nx.spring_layout(self.network_graph)
        
        nx.draw(self.network_graph, self.network_pos, ax=self.ax_network, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        
        self.ax_network.set_title("Network Connections Graph")
        self.canvas_network.draw()

    def update_threat_dashboard(self):
        self.ax_threat.clear()
        
        threat_levels = list(self.threat_levels.keys())
        counts = list(self.threat_levels.values())
        colors = ['green', 'yellow', 'red']
        
        self.ax_threat.bar(threat_levels, counts, color=colors)
        self.ax_threat.set_title("Threat Level Distribution")
        self.ax_threat.set_xlabel("Threat Level")
        self.ax_threat.set_ylabel("Count")
        
        for i, v in enumerate(counts):
            self.ax_threat.text(i, v, str(v), ha='center', va='bottom')
        
        self.canvas_threat.draw()

    def update_traffic_over_time(self):
        self.ax_traffic.clear()
        
        times = list(self.traffic_over_time.keys())
        traffic = list(self.traffic_over_time.values())
        
        self.ax_traffic.plot(times, traffic, marker='o')
        self.ax_traffic.set_title("Traffic Over Time")
        self.ax_traffic.set_xlabel("Time")
        self.ax_traffic.set_ylabel("Traffic Volume (bytes)")
        self.ax_traffic.tick_params(axis='x', rotation=45)
        
        self.canvas_traffic.draw()

    def update_anomaly_chart(self):
        self.ax_anomaly.clear()
        if self.anomalies:
            anomaly_times = [a[0] for a in self.anomalies]
            anomaly_scores = [float(a[5].split(":")[1].strip()[:-1]) for a in self.anomalies]
            threat_levels = [a[6] for a in self.anomalies]
            
            # Farklı tehdit seviyeleri için renkler
            colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Normal': 'blue'}
            
            for time, score, level in zip(anomaly_times, anomaly_scores, threat_levels):
                self.ax_anomaly.scatter(time, score, c=colors[level], label=level, s=50, edgecolors='black')
            
            self.ax_anomaly.set_title("Anomaly Scores Over Time")
            self.ax_anomaly.set_xlabel("Time")
            self.ax_anomaly.set_ylabel("Anomaly Score")
            self.ax_anomaly.tick_params(axis='x', rotation=45)
            
            # Eşik değerini göster
            self.ax_anomaly.axhline(y=self.ais.anomaly_threshold, color='r', linestyle='--', label='Anomaly Threshold')
            
            # Izgara ekle
            self.ax_anomaly.grid(True, linestyle='--', alpha=0.7)
            
            # Tekrarlanan etiketleri kaldır ve göster
            handles, labels = self.ax_anomaly.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.ax_anomaly.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
            
            # Son 10 veri noktasını göster (eğer varsa)
            if len(anomaly_times) > 10:
                self.ax_anomaly.set_xlim(anomaly_times[-10], anomaly_times[-1])
                self.ax_anomaly.set_ylim(0, max(max(anomaly_scores[-10:]), self.ais.anomaly_threshold) * 1.1)
        else:
            self.ax_anomaly.text(0.5, 0.5, "No anomalies detected", ha='center', va='center')
        
        self.canvas_anomaly.draw()

    def update_ip_connections_chart(self):
        self.ax_ip_connections.clear()
        ip_counts = {ip: len(connections) for ip, connections in self.ip_connections.items()}
        ips = list(ip_counts.keys())
        counts = list(ip_counts.values())
        self.ax_ip_connections.bar(ips, counts)
        self.ax_ip_connections.set_title("IP Connection Counts")
        self.ax_ip_connections.set_xlabel("IP Address")
        self.ax_ip_connections.set_ylabel("Connection Count")
        self.ax_ip_connections.tick_params(axis='x', rotation=90)
        self.canvas_ip_connections.draw()

    def train_ais(self):
        if len(self.packets) < self.min_packets_for_training:
            self.update_status(f"Insufficient packets for training. Current: {len(self.packets)}, Required: {self.min_packets_for_training}")
            self.ais_status_label.configure(text=f"AIS Status: Insufficient data ({len(self.packets)}/{self.min_packets_for_training})")
            return

        data = self.get_training_data()
        self.update_status(f"Training data shape: {data.shape}")
        
        if data.shape[0] == 0:
            self.update_status("No valid data for training. Collecting more packets...")
            self.ais_status_label.configure(text="AIS Status: No valid data")
            return
        
        try:
            self.update_status("Advanced AIS training started.")
            self.ais.train(data)
            self.update_status("Advanced AIS training completed.")
            self.ais_status_label.configure(text="AIS Status: Trained")
            self.show_ais_metrics()
        except ValueError as ve:
            self.update_status(f"ValueError during AIS training: {str(ve)}")
            self.ais_status_label.configure(text="AIS Status: Training error (ValueError)")
            self.master.after(0, lambda: traceback.print_exc())
        except Exception as e:
            self.update_status(f"Unexpected error during AIS training: {str(e)}")
            self.ais_status_label.configure(text="AIS Status: Training error (Unexpected)")
            self.master.after(0, lambda: traceback.print_exc())

    def get_training_data(self):
        self.update_status(f"Total captured packets: {len(self.packets)}")
        data = []
        for i, packet in enumerate(self.packets):
            if scapy.IP in packet:
                proto = packet[scapy.IP].proto
                length = len(packet)
                payload_entropy = self.calculate_entropy(packet)
                data.append([proto, length, payload_entropy])
                if i % 1 == 0 and i > 0:
                    self.update_status(f"Processed packet: {i}")
        
        self.update_status(f"Total processed packets: {len(data)}")
        
        if not data:
            self.update_status("No data found for training.")
            return np.array([]).reshape(0, 3)  # Empty 2D array with 3 features
        
        return np.array(data)

    def show_ais_metrics(self):
        if not self.ais.is_trained():
            self.ais_metrics_text.delete("1.0", tk.END)
            self.ais_metrics_text.insert(tk.END, "AIS Metrics: Not yet trained")
            return

        metrics_text = f"""AIS Metrics:
        K-Means Clustering:
        - Number of Clusters: {self.ais.kmeans_model.n_clusters}
        - Inertia: {self.ais.kmeans_model.inertia_:.2f}
        - Iterations: {self.ais.kmeans_model.n_iter_}

        Isolation Forest:
        - Number of Estimators: {self.ais.isolation_forest.n_estimators}
        - Contamination: {self.ais.isolation_forest.contamination}

        One-Class SVM:
        - Kernel: {self.ais.one_class_svm.kernel}
        - Nu: {self.ais.one_class_svm.nu}

        Clonal Selection:
        - Memory Size: {len(self.ais.clonal_selection_memory)}

        Negative Selection:
        - Number of Detectors: {len(self.ais.negative_selection_detectors)}

        Feature Scaling (StandardScaler):
        - Mean: {self.ais.scaler.mean_}
        - Variance: {self.ais.scaler.var_}
        """
        
        self.ais_metrics_text.delete("1.0", tk.END)
        self.ais_metrics_text.insert(tk.END, metrics_text)

    def generate_report(self):
        try:
            self.update_status("Starting report generation process...")
            
            pdfmetrics.registerFont(TTFont('Arial', 'C:\\Windows\\Fonts\\arial.ttf'))
            pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:\\Windows\\Fonts\\arialbd.ttf'))
            
            filename = filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF files", "*.pdf")])
            if not filename:
                self.update_status("File selection cancelled")
                
                return
            self.update_status(f"Selected file: {filename}")

            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='CustomHeading1',
                                    parent=styles['Heading1'],
                                    fontSize=14,
                                    spaceAfter=10))

            elements = []

            elements.append(Paragraph("Network Traffic Analysis Report", styles['Heading1']))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("1. Overview", styles['Heading2']))
            elements.append(Paragraph(f"Total Packets Analyzed: {len(self.packets)}", styles['BodyText']))
            elements.append(Paragraph(f"Analysis Duration: {self.get_analysis_duration()}", styles['BodyText']))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("2. Protocol Distribution", styles['Heading2']))
            elements.append(self.create_protocol_chart())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("3. Packet Size Distribution", styles['Heading2']))
            elements.append(self.create_packet_size_chart())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("4. Top IP Connections", styles['Heading2']))
            elements.append(self.create_ip_connections_table())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("5. Traffic Over Time", styles['Heading2']))
            elements.append(self.create_traffic_over_time_chart())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("6. Anomaly Detection", styles['Heading2']))
            elements.append(Paragraph(f"Total Anomalies Detected: {len(self.anomalies)}", styles['BodyText']))
            elements.append(self.create_anomalies_table())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("7. Threat Level Distribution", styles['Heading2']))
            elements.append(self.create_threat_level_chart())
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("8. AIS Metrics", styles['Heading2']))
            elements.append(self.get_ais_metrics_table())
            elements.append(Spacer(1, 12))
            
            elements.append(Paragraph("9. Network Connections Graph", styles['Heading2']))
            network_graph_image = self.create_network_graph_for_report()
            elements.append(network_graph_image)
            elements.append(Spacer(1, 12))
            elements.append(self.create_network_analysis_table())
            elements.append(Spacer(1, 12))

            doc.build(elements)
            self.update_status(f"Report generated successfully: {filename}")
            CTkMessagebox(title="Success", message=f"Report generated successfully: {filename}", icon="check")
            
        except Exception as e:
            self.update_status(f"Error generating report: {str(e)}")
            self.master.after(0, lambda: traceback.print_exc())
            CTkMessagebox(title="Error", message=f"Error generating report: {str(e)}", icon="cancel")

    def get_analysis_duration(self):
        if not self.packets:
            return "No packets captured"
        start_time = self.packets[0].time
        end_time = self.packets[-1].time
        duration = end_time - start_time
        return f"{duration:.2f} seconds"

    def create_protocol_chart(self):
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 50
        pie.width = 100
        pie.height = 100
        pie.data = list(self.protocol_counts.values())
        pie.labels = [str(key) for key in self.protocol_counts.keys()]  # Sayısal değerleri stringe çevirin
        drawing.add(pie)
        return drawing

    def create_packet_size_chart(self):
        drawing = Drawing(400, 200)
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 300
        bc.data = [self.packet_sizes]
        bc.categoryAxis.categoryNames = [str(i) for i in range(len(self.packet_sizes))]
        drawing.add(bc)
        return drawing

    def create_ip_connections_table(self):
        data = [["Source IP", "Destination IP", "Connection Count"]]
        for src, destinations in self.ip_connections.items():
            for dst in destinations:
                data.append([src, dst, len(destinations)])
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Arial-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Arial'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        return table

    def create_traffic_over_time_chart(self):
        drawing = Drawing(400, 200)
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 300
        
        # Sadece saat bilgisini al
        times = [time.split()[1] for time in self.traffic_over_time.keys()]
        traffic = list(self.traffic_over_time.values())
        
        bc.data = [traffic]
        bc.categoryAxis.categoryNames = times
        bc.categoryAxis.labels.boxAnchor = 'ne'
        bc.categoryAxis.labels.angle = 45
        bc.categoryAxis.labels.dx = -8
        bc.categoryAxis.labels.dy = -2
        
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = max(traffic) * 1.1
        bc.valueAxis.valueStep = max(traffic) // 5
        
        bc.bars[0].fillColor = colors.lightblue
        
        drawing.add(bc)
        
        # Başlık ekle
        title = String(200, 180, 'Traffic Over Time', fontSize=16, textAnchor='middle')
        drawing.add(title)
        
        return drawing

    def create_anomalies_table(self):
        data = [["Time", "Source IP", "Destination IP", "Protocol", "Length", "Status", "Threat Level","Explanation"]]
        for a in self.anomalies[:10]:  # Sadece ilk 10 anomaliyi göster
            time, src, dst, proto, length, status, threat_level = a[:7]
            explanation = self.get_anomaly_explanation(a)
            data.append([time, src, dst, proto, length, status, threat_level, explanation])
        
        table = Table(data, colWidths=[90, 70, 75, 40, 40, 85, 50,155])
        
        style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Arial-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Arial'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]
        
        # Tehdit seviyesine göre renklendirme
        for i, row in enumerate(data[1:], start=1):
            if row[6] == "High":
                style.append(('BACKGROUND', (6, i), (6, i), colors.red))
            elif row[6] == "Medium":
                style.append(('BACKGROUND', (6, i), (6, i), colors.orange))
            elif row[6] == "Low":
                style.append(('BACKGROUND', (6, i), (6, i), colors.yellow))
        
        table.setStyle(TableStyle(style))
        return table

    def get_anomaly_explanation(self, anomaly):
        _, _, _, proto, length, status, threat_level = anomaly[:7]
        anomaly_score = float(status.split(":")[1].strip()[:-1])
        
        explanations = []
        
        if anomaly_score > 0.8:
            explanations.append("Çok yüksek anomali skoru")
        elif anomaly_score > 0.6:
            explanations.append("Yüksek anomali skoru")
        
        if length > 1500:
            explanations.append("Olağandışı büyük paket boyutu")
        elif length < 20:
            explanations.append("Olağandışı küçük paket boyutu")
        
        if proto not in [6, 17]:  # TCP veya UDP değilse
            explanations.append("Nadir görülen protokol")
        
        if not explanations:
            explanations.append("Genel anormallik tespit edildi")
        
        return ", ".join(explanations)

    def create_threat_level_chart(self):
        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 50
        pie.width = 100
        pie.height = 100
        pie.data = list(self.threat_levels.values())
        pie.labels = list(self.threat_levels.keys())
        drawing.add(pie)
        return drawing


    def get_ais_metrics_table(self):
        if not self.ais.is_trained():
            return Paragraph("AIS Metrics: Not yet trained", getSampleStyleSheet()['BodyText'])

        data = [
            ["AIS Metrics", ""],
            ["K-Means Clustering", ""],
            ["Number of Clusters", str(self.ais.kmeans_model.n_clusters)],
            ["Inertia", f"{self.ais.kmeans_model.inertia_:.2f}"],
            ["Iterations", str(self.ais.kmeans_model.n_iter_)],
            ["Isolation Forest", ""],
            ["Number of Estimators", str(self.ais.isolation_forest.n_estimators)],
            ["Contamination", str(self.ais.isolation_forest.contamination)],
            ["One-Class SVM", ""],
            ["Kernel", self.ais.one_class_svm.kernel],
            ["Nu", str(self.ais.one_class_svm.nu)],
            ["Clonal Selection", ""],
            ["Memory Size", str(len(self.ais.clonal_selection_memory))],
            ["Negative Selection", ""],
            ["Number of Detectors", str(len(self.ais.negative_selection_detectors))],
            ["Feature Scaling (StandardScaler)", ""],
            ["Mean", str(self.ais.scaler.mean_)],
            ["Variance", str(self.ais.scaler.var_)]
        ]

        table = Table(data, colWidths=[200, 300])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (0, 1), colors.lightgrey),
            ('BACKGROUND', (0, 5), (0, 5), colors.lightgrey),
            ('BACKGROUND', (0, 8), (0, 8), colors.lightgrey),
            ('BACKGROUND', (0, 11), (0, 11), colors.lightgrey),
            ('BACKGROUND', (0, 13), (0, 13), colors.lightgrey),
            ('BACKGROUND', (0, 15), (0, 15), colors.lightgrey),
        ]))

        return table

    def update_status(self, message):
        self.master.after(0, lambda: self.status_label.configure(text=f"Status: {message}"))

    def create_network_graph_for_report(self):
        plt.figure(figsize=(8, 6))
        nx.draw(self.network_graph, self.network_pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        plt.title("Network Connections Graph")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        img = Image(buf)
        img.drawHeight = 300
        img.drawWidth = 400
        
        return img

    def create_network_analysis_table(self):
        num_nodes = self.network_graph.number_of_nodes()
        num_edges = self.network_graph.number_of_edges()
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        data = [
            ["Network Analysis Metrics", "Value"],
            ["Total Number of Nodes", str(num_nodes)],
            ["Total Number of Edges", str(num_edges)],
            ["Average Degree", f"{avg_degree:.2f}"],
            ["Network Density", f"{nx.density(self.network_graph):.4f}"],
            ["Number of Connected Components", str(nx.number_connected_components(self.network_graph))],
        ]
        
        # Find nodes with highest degree
        degree_dict = dict(self.network_graph.degree())
        top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:5]
        
        data.append(["Nodes with Highest Degree", ""])
        for node in top_nodes:
            data.append(["", f"{node} ({degree_dict[node]})"])

        table = Table(data, colWidths=[250, 250])
        table.setStyle(TableStyle([
            # Header style
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            # Body style
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            # Lines
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
            ('LINEAFTER', (0, 0), (0, -1), 1, colors.black),
            # Specific styles
            ('BACKGROUND', (0, -6), (0, -1), colors.lightgrey),
            ('SPAN', (0, -6), (0, -1)),
        ]))
        
        return table
    
    def on_closing(self):
        self.is_capturing = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join()
        self.master.quit()
        self.master.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = AdvancedNetworkAnalyzer(root)
    root.mainloop()
>>>>>>> 20fa4ca0bc7b61cfb69899670f901b3cd7300b10
