import argparse
import os
import socket
import sys
import time

import joblib
import numpy as np
import torch
import torch.nn as nn


DATA_SIZE = 2 ** 17
HOST      = 'localhost'
SID       = 'SCR'
PI        = 3.14159265359

INPUT_DIM = 26   

DEFAULT_MODEL_PATH  = os.path.join('output', 'driving_mlp_best.pth')
DEFAULT_SCALER_PATH = os.path.join('output', 'scaler.joblib')

GEAR_UP_RPM   = 7000
GEAR_DOWN_RPM = 3000
MAX_GEAR      = 6

MAX_STEER_DELTA          = 0.05   
MAX_STEER_DELTA_RECOVERY = 0.15   
BRAKE_DEADZONE           = 0.05   
STUCK_SPEED_THRESHOLD    = 2.0    
STUCK_STEPS              = 250    


SOFT_CORRECTION_START    = 0.75  
SOFT_CORRECTION_GAIN     = 1.6   


EDGE_BRAKE_TRACKPOS      = 0.70
EDGE_BRAKE_MIN_SPEED     = 100.0  
EDGE_BRAKE_MAX           = 0.35   


TCS_SPIN_THRESHOLD       = 8.0 
TCS_ACCEL_CLAMP          = 0.1
HARD_STEER_THRESHOLD     = 0.3
HARD_STEER_ACCEL_CLAMP   = 0.5
OFF_TRACK_ACCEL_CLAMP    = 0.3
ABS_MIN_BRAKE            = 0.1
ABS_STEER_COEFF          = 1.5


LOG_EVERY = 500   


class DrivingMLP(nn.Module):

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128),       nn.ReLU(inplace=True),
            nn.Linear(128, 64),        nn.ReLU(inplace=True),
        )
        self.steer_head = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.pedal_head = nn.Sequential(nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        steer    = self.steer_head(features)
        pedals   = self.pedal_head(features)
        return torch.cat([steer, pedals], dim=1)


def clip(v, lo, hi):
    return max(lo, min(hi, v))

def destringify(s):
    if not s:
        return s
    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(s, list):
        if len(s) < 2:
            return destringify(s[0])
        return [destringify(i) for i in s]

class ServerState:
    def __init__(self):
        self.d = {}

    def parse_server_str(self, server_string: str):
        cleaned = server_string.strip()[:-1]
        parts   = cleaned.strip().lstrip('(').rstrip(')').split(')(')
        for part in parts:
            tokens = part.split(' ')
            self.d[tokens[0]] = destringify(tokens[1:])


class DriverAction:
    def __init__(self):
        self.d = {
            'accel': 0.0, 'brake': 0.0, 'clutch': 0.0,
            'gear':  1,   'steer': 0.0,
            'focus': [-90, -45, 0, 45, 90], 'meta': 0,
        }

    def clip_to_limits(self):
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'],  0, 1)
        self.d['accel'] = clip(self.d['accel'],  0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = ''
        for k, v in self.d.items():
            out += '(' + k + ' '
            if isinstance(v, list):
                out += ' '.join(str(x) for x in v)
            else:
                out += '%.3f' % v
            out += ')'
        return out


class Client:
    def __init__(self, port: int = 3001):
        self.host, self.port, self.sid = HOST, port, SID
        self.S, self.R = ServerState(), DriverAction()
        self.so = None
        self._setup_connection()

    def _setup_connection(self):
        self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.so.settimeout(1)
        init_msg = ('%s(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 '
                    '.5 1 1.7 2.5 4 7 12 19 45)' % self.sid)
        while True:
            try:
                self.so.sendto(init_msg.encode(), (self.host, self.port))
                buf, _ = self.so.recvfrom(DATA_SIZE)
                if '***identified***' in buf.decode('utf-8'):
                    return
            except socket.error:
                time.sleep(1)

    def get_input(self) -> bool:
        while True:
            try:
                buf, _ = self.so.recvfrom(DATA_SIZE)
                text = buf.decode('utf-8')
                if '***shutdown***' in text or '***restart***' in text:
                    return False
                self.S.parse_server_str(text)
                if 'angle' in self.S.d:
                    return True
            except socket.error:
                continue

    def send(self):
        self.so.sendto(repr(self.R).encode(), (self.host, self.port))

    def close(self):
        if self.so is not None:
            self.so.close()
            self.so = None


def build_features(S: dict, scaler) -> np.ndarray:
    
    track = list(S['track'])           
    vec = np.array([[
        S['angle'], S['speedX'], S['speedY'], S['speedZ'],
        S['trackPos'], S['rpm'], S['gear'],
    ] + track], dtype=np.float32)
    return scaler.transform(vec).astype(np.float32)


def smart_gear(current_gear, rpm: float) -> int:
    
    gear = int(current_gear) if current_gear is not None else 1
    if gear <= 0:
        return 1
    if rpm > GEAR_UP_RPM and gear < MAX_GEAR:
        return gear + 1
    if rpm < GEAR_DOWN_RPM and gear > 1:
        return gear - 1
    return gear


def rate_limit_steer(target: float, prev: float, max_delta: float) -> float:
    delta = clip(target - prev, -max_delta, max_delta)
    return prev + delta


def off_track_steer_target(S: dict) -> float:
    
    return -np.sign(S['trackPos']) * 0.5 + S['angle'] * 10.0 / PI


def apply_safety_net(S: dict, R: dict):
    
    if 'wheelSpinVel' in S:
        try:
            ws        = S['wheelSpinVel']
            rear_avg  = (ws[2] + ws[3]) / 2.0
            front_avg = (ws[0] + ws[1]) / 2.0
            if rear_avg - front_avg > TCS_SPIN_THRESHOLD:
                R['accel'] = min(R['accel'], TCS_ACCEL_CLAMP)
        except (TypeError, IndexError):
            pass  

    if abs(R['steer']) > HARD_STEER_THRESHOLD:
        R['accel'] = min(R['accel'], HARD_STEER_ACCEL_CLAMP)

    max_brake  = max(ABS_MIN_BRAKE, 1.0 - ABS_STEER_COEFF * abs(R['steer']))
    R['brake'] = min(R['brake'], max_brake)

    if abs(S['trackPos']) > 0.9:
        R['accel'] = min(R['accel'], OFF_TRACK_ACCEL_CLAMP)


def handle_stuck(S: dict, R: dict, stuck_counter: int, step: int):
    
    if step < 100:
        return False, 0

    if S['speedX'] < STUCK_SPEED_THRESHOLD:
        stuck_counter += 1
        if stuck_counter > STUCK_STEPS:
            R['steer']  = -1.0 if S['trackPos'] > 0 else 1.0
            R['accel']  = 0.5
            R['brake']  = 0.0
            R['gear']   = -1
            R['clutch'] = 0.0
            return True, stuck_counter
        return False, stuck_counter

    return False, 0


def parse_args():
    p = argparse.ArgumentParser(description="TORCS Behavioral Cloning Driver.")
    p.add_argument('--port', type=int, default=3001,
                   help="Porta UDP di TORCS (default: 3001).")
    p.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                   help=f"Path al file .pth dei pesi (default: {DEFAULT_MODEL_PATH}).")
    p.add_argument('--scaler', type=str, default=DEFAULT_SCALER_PATH,
                   help=f"Path al file .joblib dello scaler (default: {DEFAULT_SCALER_PATH}).")
    p.add_argument('--raw', action='store_true',
                   help="Disabilita la safety net (TCS/ABS/anti-sovrasterzo). "
                        "Restano attivi solo stuck-recovery, rate-limit sterzo e "
                        "override off-track. Utile per valutare il modello puro.")
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in [(args.model, 'modello'), (args.scaler, 'scaler')]:
        if not os.path.isfile(path):
            print(f"[ERRORE] File {label} non trovato: {path}")
            print("        Hai lanciato train.py? Controlla la cartella di output.")
            sys.exit(1)

    scaler = joblib.load(args.scaler)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DrivingMLP(input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print(f"[INIT] Device:     {device}")
    print(f"[INIT] Modello:    {args.model}")
    print(f"[INIT] Scaler:     {args.scaler}")
    print(f"[INIT] Safety net: {'OFF (modalita RAW)' if args.raw else 'ON'}")

    client = Client(port=args.port)
    print(f"[INIT] Connesso a TORCS su porta {args.port}.\n[INIT] Guida in corso...\n")

    step          = 0
    stuck_counter = 0
    prev_steer    = 0.0

    try:
        while True:
            if not client.get_input():
                print("[STOP] TORCS ha inviato shutdown/restart.")
                break

            S = client.S.d
            R = client.R.d
            step += 1

            is_stuck, stuck_counter = handle_stuck(S, R, stuck_counter, step)
            if is_stuck:
                prev_steer = 0.0
                client.send()
                continue

            x = build_features(S, scaler)
            with torch.no_grad():
                x_t  = torch.from_numpy(x).to(device)
                pred = model(x_t).cpu().numpy()[0]   

            abs_pos = abs(S['trackPos'])
            model_steer = float(pred[0])

            if abs_pos > 1.0:
                target_steer = off_track_steer_target(S)
                max_delta    = MAX_STEER_DELTA_RECOVERY
            elif abs_pos > SOFT_CORRECTION_START:
                blend = (abs_pos - SOFT_CORRECTION_START) / (1.0 - SOFT_CORRECTION_START)
                blend = clip(blend, 0.0, 1.0)
                recovery_steer = SOFT_CORRECTION_GAIN * off_track_steer_target(S)
                target_steer   = (1.0 - blend) * model_steer + blend * recovery_steer
                max_delta      = MAX_STEER_DELTA + blend * (MAX_STEER_DELTA_RECOVERY - MAX_STEER_DELTA)
            else:
                target_steer = model_steer
                max_delta    = MAX_STEER_DELTA

            prev_steer = rate_limit_steer(target_steer, prev_steer, max_delta)
            R['steer'] = prev_steer

            R['accel'] = float(pred[1])
            R['brake'] = float(pred[2])
            if R['brake'] < BRAKE_DEADZONE:
                R['brake'] = 0.0

      
            if abs_pos > EDGE_BRAKE_TRACKPOS and S['speedX'] > EDGE_BRAKE_MIN_SPEED:
                edge_factor = (abs_pos - EDGE_BRAKE_TRACKPOS) / (1.0 - EDGE_BRAKE_TRACKPOS)
                edge_factor = clip(edge_factor, 0.0, 1.0)
                edge_brake  = edge_factor * EDGE_BRAKE_MAX
                R['brake']  = max(R['brake'], edge_brake)
                R['accel']  = R['accel'] * (1.0 - edge_factor)

            if not args.raw:
                apply_safety_net(S, R)

            R['gear'] = smart_gear(S.get('gear', 1), S['rpm'])

            client.send()

            if step % LOG_EVERY == 0:
                print(f"[STEP {step:>6d}]  "
                      f"speed={S['speedX']:6.1f}  "
                      f"trackPos={S['trackPos']:+5.2f}  "
                      f"steer={R['steer']:+5.2f}  "
                      f"accel={R['accel']:.2f}  "
                      f"brake={R['brake']:.2f}  "
                      f"gear={R['gear']}  "
                      f"rpm={S['rpm']:6.0f}")

    except KeyboardInterrupt:
        print("\n[STOP] Interrotto dall'utente.")
    finally:
        client.close()
        print("[STOP] Connessione chiusa.")


if __name__ == "__main__":
    main()