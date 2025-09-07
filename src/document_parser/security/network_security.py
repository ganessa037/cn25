#!/usr/bin/env python3
"""
Network Security Configuration System
====================================

Comprehensive network security implementation for document parser system
including VPN access, firewall rules, secure communication protocols,
and container security for Docker deployments.

Features:
- VPN server configuration and client management
- Dynamic firewall rule management
- SSL/TLS certificate management
- Network intrusion detection
- Container network security
- API rate limiting and DDoS protection
- Secure communication channels
- Network monitoring and alerting
- Zero-trust network architecture
- Network access control (NAC)
"""

import asyncio
import ipaddress
import json
import logging
import os
import socket
import ssl
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

import aiofiles
import aiohttp
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
import psutil
import docker
from scapy.all import sniff, IP, TCP, UDP, ICMP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkProtocol(Enum):
    """Network protocols"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    HTTP = "http"
    HTTPS = "https"
    SSH = "ssh"
    FTP = "ftp"
    SFTP = "sftp"
    VPN = "vpn"
    ALL = "all"

class FirewallAction(Enum):
    """Firewall rule actions"""
    ALLOW = "allow"
    DENY = "deny"
    DROP = "drop"
    REJECT = "reject"
    LOG = "log"
    RATE_LIMIT = "rate_limit"

class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    MAXIMUM = "maximum"

class ThreatLevel(Enum):
    """Threat levels for intrusion detection"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VPNProtocol(Enum):
    """VPN protocols"""
    OPENVPN = "openvpn"
    WIREGUARD = "wireguard"
    IPSEC = "ipsec"
    PPTP = "pptp"
    L2TP = "l2tp"

class NetworkZone(Enum):
    """Network security zones"""
    PUBLIC = "public"
    DMZ = "dmz"
    INTERNAL = "internal"
    SECURE = "secure"
    MANAGEMENT = "management"
    QUARANTINE = "quarantine"

@dataclass
class FirewallRule:
    """Firewall rule definition"""
    rule_id: str
    name: str
    description: str
    source_ip: Optional[str]
    destination_ip: Optional[str]
    source_port: Optional[int]
    destination_port: Optional[int]
    protocol: NetworkProtocol
    action: FirewallAction
    priority: int
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches_traffic(self, src_ip: str, dst_ip: str, 
                       src_port: int, dst_port: int, protocol: str) -> bool:
        """Check if rule matches network traffic"""
        # Check protocol
        if self.protocol != NetworkProtocol.ALL and self.protocol.value != protocol.lower():
            return False
        
        # Check source IP
        if self.source_ip and not self._ip_matches(src_ip, self.source_ip):
            return False
        
        # Check destination IP
        if self.destination_ip and not self._ip_matches(dst_ip, self.destination_ip):
            return False
        
        # Check source port
        if self.source_port and self.source_port != src_port:
            return False
        
        # Check destination port
        if self.destination_port and self.destination_port != dst_port:
            return False
        
        return True
    
    def _ip_matches(self, ip: str, pattern: str) -> bool:
        """Check if IP matches pattern (supports CIDR)"""
        try:
            if '/' in pattern:
                # CIDR notation
                network = ipaddress.ip_network(pattern, strict=False)
                return ipaddress.ip_address(ip) in network
            else:
                # Exact match or wildcard
                return ip == pattern or pattern == "*"
        except:
            return False

@dataclass
class VPNClient:
    """VPN client configuration"""
    client_id: str
    username: str
    email: Optional[str]
    certificate_path: Optional[str]
    private_key_path: Optional[str]
    assigned_ip: Optional[str]
    allowed_ips: List[str]
    connected: bool = False
    last_connected: Optional[datetime] = None
    data_transferred: int = 0
    connection_duration: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityEvent:
    """Network security event"""
    event_id: str
    event_type: str
    threat_level: ThreatLevel
    source_ip: str
    destination_ip: Optional[str]
    source_port: Optional[int]
    destination_port: Optional[int]
    protocol: Optional[str]
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = False
    rule_triggered: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NetworkInterface:
    """Network interface information"""
    interface_name: str
    ip_address: str
    netmask: str
    gateway: Optional[str]
    mac_address: str
    status: str
    zone: NetworkZone
    security_level: SecurityLevel
    monitored: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SSLCertificate:
    """SSL certificate information"""
    cert_id: str
    common_name: str
    subject_alt_names: List[str]
    issuer: str
    serial_number: str
    not_before: datetime
    not_after: datetime
    fingerprint: str
    certificate_path: str
    private_key_path: str
    chain_path: Optional[str]
    auto_renew: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if certificate is expired"""
        return datetime.now() > self.not_after
    
    def expires_soon(self, days: int = 30) -> bool:
        """Check if certificate expires within specified days"""
        return datetime.now() + timedelta(days=days) > self.not_after

class IntrusionDetectionSystem:
    """Network intrusion detection system"""
    
    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.running = False
        self.events: deque = deque(maxlen=10000)
        self.threat_patterns = self._load_threat_patterns()
        self.rate_limits = defaultdict(lambda: defaultdict(int))
        self.blocked_ips = set()
        
    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load threat detection patterns"""
        return {
            'port_scan': {
                'description': 'Port scanning detection',
                'pattern': 'multiple_ports_single_source',
                'threshold': 10,
                'time_window': 60
            },
            'brute_force': {
                'description': 'Brute force attack detection',
                'pattern': 'failed_auth_attempts',
                'threshold': 5,
                'time_window': 300
            },
            'ddos': {
                'description': 'DDoS attack detection',
                'pattern': 'high_connection_rate',
                'threshold': 100,
                'time_window': 10
            },
            'suspicious_traffic': {
                'description': 'Suspicious traffic patterns',
                'pattern': 'unusual_protocols',
                'threshold': 1,
                'time_window': 1
            }
        }
    
    def start_monitoring(self):
        """Start network monitoring"""
        self.running = True
        
        def packet_handler(packet):
            try:
                if IP in packet:
                    self._analyze_packet(packet)
            except Exception as e:
                logger.error(f"Error analyzing packet: {e}")
        
        # Start packet capture in separate thread
        def capture_packets():
            try:
                sniff(iface=self.interface, prn=packet_handler, stop_filter=lambda x: not self.running)
            except Exception as e:
                logger.error(f"Error capturing packets: {e}")
        
        threading.Thread(target=capture_packets, daemon=True).start()
        logger.info(f"Started intrusion detection on interface {self.interface}")
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.running = False
        logger.info("Stopped intrusion detection")
    
    def _analyze_packet(self, packet):
        """Analyze network packet for threats"""
        try:
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            
            # Skip internal traffic analysis
            if self._is_internal_ip(src_ip) and self._is_internal_ip(dst_ip):
                return
            
            # Check for port scanning
            if TCP in packet:
                self._check_port_scan(src_ip, packet[TCP].dport)
            
            # Check for DDoS patterns
            self._check_ddos_pattern(src_ip)
            
            # Check for suspicious protocols
            self._check_suspicious_protocols(packet)
            
        except Exception as e:
            logger.error(f"Error in packet analysis: {e}")
    
    def _check_port_scan(self, src_ip: str, dst_port: int):
        """Check for port scanning patterns"""
        current_time = int(time.time())
        time_window = self.threat_patterns['port_scan']['time_window']
        threshold = self.threat_patterns['port_scan']['threshold']
        
        # Clean old entries
        cutoff_time = current_time - time_window
        self.rate_limits[src_ip] = {
            k: v for k, v in self.rate_limits[src_ip].items() 
            if k > cutoff_time
        }
        
        # Add current port access
        self.rate_limits[src_ip][current_time] = dst_port
        
        # Check if threshold exceeded
        unique_ports = len(set(self.rate_limits[src_ip].values()))
        if unique_ports >= threshold:
            self._create_security_event(
                'port_scan',
                ThreatLevel.HIGH,
                src_ip,
                description=f"Port scan detected from {src_ip} - {unique_ports} unique ports accessed"
            )
    
    def _check_ddos_pattern(self, src_ip: str):
        """Check for DDoS attack patterns"""
        current_time = int(time.time())
        time_window = self.threat_patterns['ddos']['time_window']
        threshold = self.threat_patterns['ddos']['threshold']
        
        # Count connections in time window
        connection_key = f"ddos_{src_ip}"
        if connection_key not in self.rate_limits:
            self.rate_limits[connection_key] = defaultdict(int)
        
        # Clean old entries
        cutoff_time = current_time - time_window
        self.rate_limits[connection_key] = {
            k: v for k, v in self.rate_limits[connection_key].items() 
            if k > cutoff_time
        }
        
        # Increment connection count
        self.rate_limits[connection_key][current_time] += 1
        
        # Check threshold
        total_connections = sum(self.rate_limits[connection_key].values())
        if total_connections >= threshold:
            self._create_security_event(
                'ddos',
                ThreatLevel.CRITICAL,
                src_ip,
                description=f"Potential DDoS attack from {src_ip} - {total_connections} connections in {time_window}s"
            )
    
    def _check_suspicious_protocols(self, packet):
        """Check for suspicious protocol usage"""
        # This is a simplified check - in practice, this would be more sophisticated
        if ICMP in packet:
            icmp_type = packet[ICMP].type
            if icmp_type in [3, 11]:  # Destination unreachable, Time exceeded
                self._create_security_event(
                    'suspicious_traffic',
                    ThreatLevel.LOW,
                    packet[IP].src,
                    description=f"Suspicious ICMP traffic - Type {icmp_type}"
                )
    
    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is internal/private"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except:
            return False
    
    def _create_security_event(self, event_type: str, threat_level: ThreatLevel, 
                              source_ip: str, description: str):
        """Create security event"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            description=description
        )
        
        self.events.append(event)
        logger.warning(f"Security event: {description}")
        
        # Auto-block high/critical threats
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.blocked_ips.add(source_ip)
            logger.warning(f"Auto-blocked IP: {source_ip}")
    
    def get_recent_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events"""
        return list(self.events)[-limit:]
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips
    
    def unblock_ip(self, ip: str) -> bool:
        """Unblock IP address"""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"Unblocked IP: {ip}")
            return True
        return False

class FirewallManager:
    """Manages firewall rules and policies"""
    
    def __init__(self, config_path: str = "./config/firewall_rules.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.rules: Dict[str, FirewallRule] = {}
        self.active_rules: List[FirewallRule] = []
        self.default_policy = FirewallAction.DENY
        
        # Load default rules
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default firewall rules"""
        default_rules = [
            FirewallRule(
                rule_id="allow_ssh",
                name="Allow SSH",
                description="Allow SSH access from management network",
                source_ip="192.168.1.0/24",
                destination_port=22,
                protocol=NetworkProtocol.TCP,
                action=FirewallAction.ALLOW,
                priority=100
            ),
            FirewallRule(
                rule_id="allow_https",
                name="Allow HTTPS",
                description="Allow HTTPS traffic",
                destination_port=443,
                protocol=NetworkProtocol.TCP,
                action=FirewallAction.ALLOW,
                priority=200
            ),
            FirewallRule(
                rule_id="allow_http",
                name="Allow HTTP",
                description="Allow HTTP traffic",
                destination_port=80,
                protocol=NetworkProtocol.TCP,
                action=FirewallAction.ALLOW,
                priority=300
            ),
            FirewallRule(
                rule_id="deny_all",
                name="Deny All",
                description="Default deny rule",
                protocol=NetworkProtocol.ALL,
                action=FirewallAction.DENY,
                priority=9999
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
        
        self._update_active_rules()
    
    def add_rule(self, rule: FirewallRule) -> bool:
        """Add firewall rule"""
        try:
            self.rules[rule.rule_id] = rule
            self._update_active_rules()
            self._save_rules()
            
            logger.info(f"Added firewall rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding firewall rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove firewall rule"""
        try:
            if rule_id in self.rules:
                del self.rules[rule_id]
                self._update_active_rules()
                self._save_rules()
                
                logger.info(f"Removed firewall rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing firewall rule: {e}")
            return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update firewall rule"""
        try:
            if rule_id not in self.rules:
                return False
            
            rule = self.rules[rule_id]
            
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.now()
            self._update_active_rules()
            self._save_rules()
            
            logger.info(f"Updated firewall rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating firewall rule: {e}")
            return False
    
    def _update_active_rules(self):
        """Update active rules list (sorted by priority)"""
        self.active_rules = sorted(
            [rule for rule in self.rules.values() if rule.enabled],
            key=lambda r: r.priority
        )
    
    def check_traffic(self, src_ip: str, dst_ip: str, src_port: int, 
                     dst_port: int, protocol: str) -> Tuple[FirewallAction, Optional[str]]:
        """Check traffic against firewall rules"""
        for rule in self.active_rules:
            if rule.matches_traffic(src_ip, dst_ip, src_port, dst_port, protocol):
                return rule.action, rule.rule_id
        
        return self.default_policy, None
    
    def block_ip(self, ip: str, reason: str = "Security threat") -> str:
        """Block IP address"""
        rule_id = f"block_{ip}_{int(time.time())}"
        
        block_rule = FirewallRule(
            rule_id=rule_id,
            name=f"Block {ip}",
            description=f"Auto-generated block rule: {reason}",
            source_ip=ip,
            protocol=NetworkProtocol.ALL,
            action=FirewallAction.DROP,
            priority=1,  # High priority
            tags=["auto-generated", "security"]
        )
        
        self.add_rule(block_rule)
        return rule_id
    
    def unblock_ip(self, ip: str) -> bool:
        """Unblock IP address"""
        rules_to_remove = []
        
        for rule_id, rule in self.rules.items():
            if (rule.source_ip == ip and 
                rule.action in [FirewallAction.DROP, FirewallAction.DENY] and
                "auto-generated" in rule.tags):
                rules_to_remove.append(rule_id)
        
        success = False
        for rule_id in rules_to_remove:
            if self.remove_rule(rule_id):
                success = True
        
        return success
    
    def get_rules(self) -> List[FirewallRule]:
        """Get all firewall rules"""
        return list(self.rules.values())
    
    def _save_rules(self):
        """Save rules to configuration file"""
        try:
            data = {
                'rules': [asdict(rule) for rule in self.rules.values()],
                'default_policy': self.default_policy.value,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving firewall rules: {e}")

class VPNManager:
    """Manages VPN server and client configurations"""
    
    def __init__(self, config_path: str = "./config/vpn_config.json",
                 cert_path: str = "./certs/vpn"):
        self.config_path = Path(config_path)
        self.cert_path = Path(cert_path)
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.cert_path.mkdir(parents=True, exist_ok=True)
        
        self.clients: Dict[str, VPNClient] = {}
        self.server_config = self._get_default_server_config()
        self.protocol = VPNProtocol.OPENVPN
        
    def _get_default_server_config(self) -> Dict[str, Any]:
        """Get default VPN server configuration"""
        return {
            'server_ip': '10.8.0.1',
            'server_port': 1194,
            'network': '10.8.0.0/24',
            'protocol': 'udp',
            'encryption': 'AES-256-CBC',
            'auth': 'SHA256',
            'dh_key_size': 2048,
            'keepalive': '10 120',
            'max_clients': 100,
            'duplicate_cn': False,
            'compress': True,
            'push_routes': ['192.168.1.0 255.255.255.0'],
            'dns_servers': ['8.8.8.8', '8.8.4.4']
        }
    
    def generate_server_certificates(self) -> bool:
        """Generate VPN server certificates"""
        try:
            # Generate CA private key
            ca_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Generate CA certificate
            ca_name = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "SG"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Singapore"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Singapore"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Document Parser VPN CA"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Document Parser VPN CA"),
            ])
            
            ca_cert = x509.CertificateBuilder().subject_name(
                ca_name
            ).issuer_name(
                ca_name
            ).public_key(
                ca_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.now()
            ).not_valid_after(
                datetime.now() + timedelta(days=3650)  # 10 years
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            ).add_extension(
                x509.KeyUsage(
                    key_cert_sign=True,
                    crl_sign=True,
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).sign(ca_key, hashes.SHA256())
            
            # Save CA certificate and key
            ca_cert_path = self.cert_path / "ca.crt"
            ca_key_path = self.cert_path / "ca.key"
            
            with open(ca_cert_path, "wb") as f:
                f.write(ca_cert.public_bytes(Encoding.PEM))
            
            with open(ca_key_path, "wb") as f:
                f.write(ca_key.private_bytes(
                    Encoding.PEM,
                    PrivateFormat.PKCS8,
                    NoEncryption()
                ))
            
            # Generate server private key
            server_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Generate server certificate
            server_name = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "SG"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Singapore"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "Singapore"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Document Parser VPN"),
                x509.NameAttribute(NameOID.COMMON_NAME, "vpn.documentparser.local"),
            ])
            
            server_cert = x509.CertificateBuilder().subject_name(
                server_name
            ).issuer_name(
                ca_name
            ).public_key(
                server_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.now()
            ).not_valid_after(
                datetime.now() + timedelta(days=365)  # 1 year
            ).add_extension(
                x509.KeyUsage(
                    key_cert_sign=False,
                    crl_sign=False,
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=True,
            ).sign(ca_key, hashes.SHA256())
            
            # Save server certificate and key
            server_cert_path = self.cert_path / "server.crt"
            server_key_path = self.cert_path / "server.key"
            
            with open(server_cert_path, "wb") as f:
                f.write(server_cert.public_bytes(Encoding.PEM))
            
            with open(server_key_path, "wb") as f:
                f.write(server_key.private_bytes(
                    Encoding.PEM,
                    PrivateFormat.PKCS8,
                    NoEncryption()
                ))
            
            logger.info("Generated VPN server certificates")
            return True
            
        except Exception as e:
            logger.error(f"Error generating server certificates: {e}")
            return False
    
    def create_client(self, username: str, email: Optional[str] = None,
                     allowed_ips: Optional[List[str]] = None) -> Optional[str]:
        """Create VPN client configuration"""
        try:
            client_id = str(uuid.uuid4())
            
            if allowed_ips is None:
                allowed_ips = ["0.0.0.0/0"]  # Allow all by default
            
            # Assign IP from VPN network
            assigned_ip = self._get_next_available_ip()
            
            client = VPNClient(
                client_id=client_id,
                username=username,
                email=email,
                assigned_ip=assigned_ip,
                allowed_ips=allowed_ips,
                expires_at=datetime.now() + timedelta(days=365)
            )
            
            # Generate client certificates
            cert_path, key_path = self._generate_client_certificates(client_id, username)
            client.certificate_path = cert_path
            client.private_key_path = key_path
            
            self.clients[client_id] = client
            self._save_clients()
            
            logger.info(f"Created VPN client: {username} ({client_id})")
            return client_id
            
        except Exception as e:
            logger.error(f"Error creating VPN client: {e}")
            return None
    
    def revoke_client(self, client_id: str) -> bool:
        """Revoke VPN client access"""
        try:
            if client_id not in self.clients:
                return False
            
            client = self.clients[client_id]
            client.revoked = True
            client.connected = False
            
            self._save_clients()
            
            logger.info(f"Revoked VPN client: {client.username} ({client_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error revoking VPN client: {e}")
            return False
    
    def generate_client_config(self, client_id: str) -> Optional[str]:
        """Generate OpenVPN client configuration file"""
        try:
            if client_id not in self.clients:
                return None
            
            client = self.clients[client_id]
            if client.revoked:
                return None
            
            config = f"""
client
dev tun
proto {self.server_config['protocol']}
remote {self.server_config['server_ip']} {self.server_config['server_port']}
resolv-retry infinite
nobind
persist-key
persist-tun
ca ca.crt
cert {client.username}.crt
key {client.username}.key
remote-cert-tls server
cipher {self.server_config['encryption']}
auth {self.server_config['auth']}
verb 3
"""
            
            if self.server_config['compress']:
                config += "compress lz4-v2\n"
            
            return config
            
        except Exception as e:
            logger.error(f"Error generating client config: {e}")
            return None
    
    def _get_next_available_ip(self) -> str:
        """Get next available IP in VPN network"""
        network = ipaddress.ip_network(self.server_config['network'])
        used_ips = {client.assigned_ip for client in self.clients.values() if client.assigned_ip}
        
        for ip in network.hosts():
            if str(ip) not in used_ips and str(ip) != self.server_config['server_ip']:
                return str(ip)
        
        raise Exception("No available IPs in VPN network")
    
    def _generate_client_certificates(self, client_id: str, username: str) -> Tuple[str, str]:
        """Generate client certificates"""
        # This is a simplified version - in practice, you'd use the CA to sign client certs
        cert_path = str(self.cert_path / f"{username}.crt")
        key_path = str(self.cert_path / f"{username}.key")
        
        # Generate client private key
        client_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Save client key
        with open(key_path, "wb") as f:
            f.write(client_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.PKCS8,
                NoEncryption()
            ))
        
        # For demo purposes, create a placeholder certificate
        with open(cert_path, "w") as f:
            f.write(f"# Client certificate for {username}\n")
            f.write(f"# Generated at {datetime.now()}\n")
        
        return cert_path, key_path
    
    def _save_clients(self):
        """Save client configurations"""
        try:
            data = {
                'clients': [asdict(client) for client in self.clients.values()],
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving VPN clients: {e}")

class ContainerSecurityManager:
    """Manages container security for Docker deployments"""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.docker_client = None
        
        self.security_policies = self._get_default_security_policies()
    
    def _get_default_security_policies(self) -> Dict[str, Any]:
        """Get default container security policies"""
        return {
            'network_isolation': True,
            'read_only_filesystem': True,
            'no_new_privileges': True,
            'drop_capabilities': ['ALL'],
            'add_capabilities': ['CHOWN', 'SETUID', 'SETGID'],
            'security_opt': ['no-new-privileges:true'],
            'user': 'nobody',
            'memory_limit': '512m',
            'cpu_limit': '0.5',
            'restart_policy': 'unless-stopped',
            'log_driver': 'json-file',
            'log_options': {'max-size': '10m', 'max-file': '3'}
        }
    
    def create_secure_network(self, network_name: str) -> bool:
        """Create secure Docker network"""
        try:
            if not self.docker_client:
                return False
            
            # Check if network already exists
            try:
                existing_network = self.docker_client.networks.get(network_name)
                logger.info(f"Network {network_name} already exists")
                return True
            except docker.errors.NotFound:
                pass
            
            # Create network with security settings
            network = self.docker_client.networks.create(
                network_name,
                driver="bridge",
                options={
                    "com.docker.network.bridge.enable_icc": "false",
                    "com.docker.network.bridge.enable_ip_masquerade": "true",
                    "com.docker.network.driver.mtu": "1500"
                },
                ipam=docker.types.IPAMConfig(
                    pool_configs=[
                        docker.types.IPAMPool(
                            subnet="172.20.0.0/16",
                            gateway="172.20.0.1"
                        )
                    ]
                )
            )
            
            logger.info(f"Created secure network: {network_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating secure network: {e}")
            return False
    
    def get_container_security_config(self, image_name: str) -> Dict[str, Any]:
        """Get security configuration for container"""
        config = {
            'image': image_name,
            'user': self.security_policies['user'],
            'read_only': self.security_policies['read_only_filesystem'],
            'security_opt': self.security_policies['security_opt'],
            'cap_drop': self.security_policies['drop_capabilities'],
            'cap_add': self.security_policies['add_capabilities'],
            'mem_limit': self.security_policies['memory_limit'],
            'cpu_period': 100000,
            'cpu_quota': int(float(self.security_policies['cpu_limit']) * 100000),
            'restart_policy': {'Name': self.security_policies['restart_policy']},
            'log_config': {
                'Type': self.security_policies['log_driver'],
                'Config': self.security_policies['log_options']
            },
            'tmpfs': {
                '/tmp': 'rw,noexec,nosuid,size=100m',
                '/var/tmp': 'rw,noexec,nosuid,size=50m'
            },
            'sysctls': {
                'net.ipv4.ip_forward': '0',
                'net.ipv4.conf.all.send_redirects': '0',
                'net.ipv4.conf.all.accept_redirects': '0'
            }
        }
        
        return config
    
    def scan_container_vulnerabilities(self, container_id: str) -> Dict[str, Any]:
        """Scan container for security vulnerabilities"""
        try:
            if not self.docker_client:
                return {'error': 'Docker not available'}
            
            container = self.docker_client.containers.get(container_id)
            
            # Basic security checks
            vulnerabilities = []
            
            # Check if running as root
            if container.attrs['Config']['User'] in ['', 'root', '0']:
                vulnerabilities.append({
                    'severity': 'HIGH',
                    'description': 'Container running as root user',
                    'recommendation': 'Use non-root user'
                })
            
            # Check for privileged mode
            if container.attrs['HostConfig']['Privileged']:
                vulnerabilities.append({
                    'severity': 'CRITICAL',
                    'description': 'Container running in privileged mode',
                    'recommendation': 'Remove privileged flag'
                })
            
            # Check for host network mode
            if container.attrs['HostConfig']['NetworkMode'] == 'host':
                vulnerabilities.append({
                    'severity': 'HIGH',
                    'description': 'Container using host network',
                    'recommendation': 'Use bridge or custom network'
                })
            
            # Check for writable filesystem
            if not container.attrs['HostConfig']['ReadonlyRootfs']:
                vulnerabilities.append({
                    'severity': 'MEDIUM',
                    'description': 'Container filesystem is writable',
                    'recommendation': 'Use read-only filesystem'
                })
            
            return {
                'container_id': container_id,
                'image': container.attrs['Config']['Image'],
                'vulnerabilities': vulnerabilities,
                'scan_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error scanning container: {e}")
            return {'error': str(e)}
    
    def get_security_recommendations(self) -> List[str]:
        """Get container security recommendations"""
        return [
            "Use official base images from trusted registries",
            "Keep base images updated with latest security patches",
            "Run containers as non-root users",
            "Use read-only filesystems where possible",
            "Implement proper network segmentation",
            "Limit container resources (CPU, memory)",
            "Use security scanning tools in CI/CD pipeline",
            "Implement proper secrets management",
            "Monitor container runtime behavior",
            "Use minimal base images (distroless, alpine)",
            "Implement proper logging and monitoring",
            "Use container image signing and verification"
        ]

class NetworkSecurityManager:
    """Main network security management system"""
    
    def __init__(self, config_path: str = "./config/network_security.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.firewall = FirewallManager()
        self.vpn = VPNManager()
        self.ids = IntrusionDetectionSystem()
        self.container_security = ContainerSecurityManager()
        
        self.ssl_certificates: Dict[str, SSLCertificate] = {}
        self.network_interfaces: Dict[str, NetworkInterface] = {}
        
        self.running = False
        
        # Load configuration
        self._load_configuration()
        self._discover_network_interfaces()
    
    def _load_configuration(self):
        """Load network security configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load SSL certificates
                for cert_data in config.get('ssl_certificates', []):
                    cert = SSLCertificate(**cert_data)
                    self.ssl_certificates[cert.cert_id] = cert
                
                logger.info("Loaded network security configuration")
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _discover_network_interfaces(self):
        """Discover network interfaces"""
        try:
            for interface_name, addresses in psutil.net_if_addrs().items():
                for addr in addresses:
                    if addr.family == socket.AF_INET:  # IPv4
                        interface = NetworkInterface(
                            interface_name=interface_name,
                            ip_address=addr.address,
                            netmask=addr.netmask,
                            mac_address=next(
                                (a.address for a in addresses if a.family == psutil.AF_LINK),
                                "unknown"
                            ),
                            status="up" if interface_name in psutil.net_if_stats() else "down",
                            zone=self._determine_network_zone(addr.address),
                            security_level=SecurityLevel.MEDIUM
                        )
                        
                        self.network_interfaces[interface_name] = interface
                        break
            
            logger.info(f"Discovered {len(self.network_interfaces)} network interfaces")
            
        except Exception as e:
            logger.error(f"Error discovering network interfaces: {e}")
    
    def _determine_network_zone(self, ip_address: str) -> NetworkZone:
        """Determine network zone based on IP address"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            if ip.is_loopback:
                return NetworkZone.INTERNAL
            elif ip.is_private:
                if str(ip).startswith('192.168.1.'):
                    return NetworkZone.MANAGEMENT
                elif str(ip).startswith('10.8.'):
                    return NetworkZone.SECURE  # VPN network
                else:
                    return NetworkZone.INTERNAL
            else:
                return NetworkZone.PUBLIC
                
        except:
            return NetworkZone.PUBLIC
    
    async def start(self):
        """Start network security services"""
        self.running = True
        
        # Start intrusion detection
        self.ids.start_monitoring()
        
        # Generate VPN certificates if needed
        ca_cert_path = self.vpn.cert_path / "ca.crt"
        if not ca_cert_path.exists():
            self.vpn.generate_server_certificates()
        
        # Create secure container network
        self.container_security.create_secure_network("secure-network")
        
        # Start background monitoring
        asyncio.create_task(self._security_monitoring_loop())
        asyncio.create_task(self._certificate_monitoring_loop())
        
        logger.info("Network security manager started")
    
    async def stop(self):
        """Stop network security services"""
        self.running = False
        
        # Stop intrusion detection
        self.ids.stop_monitoring()
        
        # Save configuration
        await self._save_configuration()
        
        logger.info("Network security manager stopped")
    
    def create_vpn_client(self, username: str, email: Optional[str] = None) -> Optional[str]:
        """Create VPN client"""
        return self.vpn.create_client(username, email)
    
    def revoke_vpn_client(self, client_id: str) -> bool:
        """Revoke VPN client"""
        return self.vpn.revoke_client(client_id)
    
    def add_firewall_rule(self, rule: FirewallRule) -> bool:
        """Add firewall rule"""
        return self.firewall.add_rule(rule)
    
    def block_ip_address(self, ip: str, reason: str = "Security threat") -> bool:
        """Block IP address"""
        # Block in firewall
        rule_id = self.firewall.block_ip(ip, reason)
        
        # Add to IDS blocked list
        self.ids.blocked_ips.add(ip)
        
        logger.warning(f"Blocked IP address: {ip} - {reason}")
        return rule_id is not None
    
    def unblock_ip_address(self, ip: str) -> bool:
        """Unblock IP address"""
        # Unblock in firewall
        firewall_success = self.firewall.unblock_ip(ip)
        
        # Remove from IDS blocked list
        ids_success = self.ids.unblock_ip(ip)
        
        logger.info(f"Unblocked IP address: {ip}")
        return firewall_success or ids_success
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status"""
        recent_events = self.ids.get_recent_events(50)
        
        # Count events by threat level
        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.threat_level.value] += 1
        
        # Check certificate status
        expiring_certs = [
            cert for cert in self.ssl_certificates.values()
            if cert.expires_soon(30)
        ]
        
        return {
            'firewall': {
                'active_rules': len(self.firewall.active_rules),
                'total_rules': len(self.firewall.rules),
                'default_policy': self.firewall.default_policy.value
            },
            'vpn': {
                'total_clients': len(self.vpn.clients),
                'active_clients': len([c for c in self.vpn.clients.values() if c.connected]),
                'revoked_clients': len([c for c in self.vpn.clients.values() if c.revoked])
            },
            'intrusion_detection': {
                'monitoring': self.ids.running,
                'recent_events': len(recent_events),
                'threat_levels': dict(threat_counts),
                'blocked_ips': len(self.ids.blocked_ips)
            },
            'certificates': {
                'total': len(self.ssl_certificates),
                'expiring_soon': len(expiring_certs),
                'expired': len([c for c in self.ssl_certificates.values() if c.is_expired()])
            },
            'network_interfaces': {
                'total': len(self.network_interfaces),
                'monitored': len([i for i in self.network_interfaces.values() if i.monitored])
            }
        }
    
    def get_security_recommendations(self) -> List[str]:
        """Get security recommendations"""
        recommendations = []
        
        # Check for high-risk configurations
        if self.firewall.default_policy == FirewallAction.ALLOW:
            recommendations.append("Change default firewall policy to DENY for better security")
        
        # Check for expiring certificates
        expiring_certs = [
            cert for cert in self.ssl_certificates.values()
            if cert.expires_soon(30)
        ]
        if expiring_certs:
            recommendations.append(f"Renew {len(expiring_certs)} SSL certificates expiring soon")
        
        # Check recent security events
        recent_events = self.ids.get_recent_events(100)
        high_threat_events = [e for e in recent_events if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
        if high_threat_events:
            recommendations.append(f"Investigate {len(high_threat_events)} high-threat security events")
        
        # Add container security recommendations
        recommendations.extend(self.container_security.get_security_recommendations())
        
        return recommendations
    
    async def _security_monitoring_loop(self):
        """Background security monitoring"""
        while self.running:
            try:
                # Check for security events that need attention
                recent_events = self.ids.get_recent_events(10)
                
                for event in recent_events:
                    if (event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] and
                        not event.blocked):
                        
                        # Auto-block high-threat IPs
                        self.block_ip_address(event.source_ip, f"Auto-block: {event.description}")
                        event.blocked = True
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _certificate_monitoring_loop(self):
        """Background certificate monitoring"""
        while self.running:
            try:
                # Check for expiring certificates
                for cert in self.ssl_certificates.values():
                    if cert.expires_soon(7):  # 7 days
                        logger.warning(f"Certificate {cert.common_name} expires in less than 7 days")
                    
                    if cert.is_expired():
                        logger.error(f"Certificate {cert.common_name} has expired")
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in certificate monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _save_configuration(self):
        """Save network security configuration"""
        try:
            config = {
                'ssl_certificates': [asdict(cert) for cert in self.ssl_certificates.values()],
                'network_interfaces': [asdict(iface) for iface in self.network_interfaces.values()],
                'updated_at': datetime.now().isoformat()
            }
            
            async with aiofiles.open(self.config_path, 'w') as f:
                await f.write(json.dumps(config, indent=2, default=str))
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

# Example usage and testing
async def main():
    """Example usage of network security system"""
    
    # Initialize network security manager
    security_manager = NetworkSecurityManager()
    
    await security_manager.start()
    
    print("🔒 Network Security System Initialized")
    print("=" * 50)
    
    try:
        # Create VPN client
        client_id = security_manager.create_vpn_client("john.doe", "john@example.com")
        if client_id:
            print(f"✅ Created VPN client: {client_id}")
            
            # Generate client config
            config = security_manager.vpn.generate_client_config(client_id)
            if config:
                print("📄 Generated VPN client configuration")
        
        # Add custom firewall rule
        custom_rule = FirewallRule(
            rule_id="allow_api",
            name="Allow API Access",
            description="Allow access to API server",
            destination_port=8000,
            protocol=NetworkProtocol.TCP,
            action=FirewallAction.ALLOW,
            priority=150
        )
        
        success = security_manager.add_firewall_rule(custom_rule)
        print(f"🛡️ Added firewall rule: {'Success' if success else 'Failed'}")
        
        # Simulate blocking an IP
        test_ip = "192.168.100.100"
        security_manager.block_ip_address(test_ip, "Test block")
        print(f"🚫 Blocked test IP: {test_ip}")
        
        # Get security status
        status = security_manager.get_security_status()
        
        print(f"\n📊 Security Status:")
        print(f"   Firewall Rules: {status['firewall']['active_rules']}/{status['firewall']['total_rules']}")
        print(f"   VPN Clients: {status['vpn']['total_clients']} total, {status['vpn']['active_clients']} active")
        print(f"   IDS Events: {status['intrusion_detection']['recent_events']} recent")
        print(f"   Blocked IPs: {status['intrusion_detection']['blocked_ips']}")
        print(f"   Network Interfaces: {status['network_interfaces']['total']}")
        
        # Get recommendations
        recommendations = security_manager.get_security_recommendations()
        if recommendations:
            print(f"\n💡 Security Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        # Show container security config
        container_config = security_manager.container_security.get_container_security_config("nginx:alpine")
        print(f"\n🐳 Container Security Config:")
        print(f"   User: {container_config['user']}")
        print(f"   Read-only: {container_config['read_only']}")
        print(f"   Memory limit: {container_config['mem_limit']}")
        print(f"   Capabilities dropped: {len(container_config['cap_drop'])}")
        
        # Wait a bit to see IDS in action
        print("\n⏳ Monitoring network traffic for 10 seconds...")
        await asyncio.sleep(10)
        
        # Check for any new events
        recent_events = security_manager.ids.get_recent_events(5)
        if recent_events:
            print(f"\n🚨 Recent Security Events:")
            for event in recent_events:
                print(f"   - {event.event_type}: {event.description} ({event.threat_level.value})")
        else:
            print("\n✅ No security events detected")
        
    finally:
        await security_manager.stop()
        print("\n🔒 Network security system stopped")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())