import os
import sys
import time
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from pynput.keyboard import Controller

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import snakeoil3_gym as snakeoil3

PI = 3.14159265359

dataset_dir = os.path.join(current_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)
master_file = os.path.join(dataset_dir, "bot_dataset.csv")

class RaceFinished(Exception): pass
class BotStuck(Exception): pass
class LapCompleted(Exception): pass

def check_bot_status(speed_kmh: float, track_pos: float, bot_state: Dict[str, Any]) -> bool:
    
    is_on_track = abs(track_pos) < 1.0

    if abs(speed_kmh) < 5.0:
        bot_state['stuck_frames'] = bot_state.get('stuck_frames', 0) + 1
    else:
        bot_state['stuck_frames'] = 0

    if bot_state['stuck_frames'] > 200:
        raise BotStuck("[ERROR] Car stopped for too long!")

    if not is_on_track:
        bot_state['offtrack_frames'] = bot_state.get('offtrack_frames', 0) + 1
    else:
        bot_state['offtrack_frames'] = 0

    if bot_state['offtrack_frames'] > 100:
        raise BotStuck("[ERROR] Off track for too long!")
        
    return is_on_track

def calculate_steering(angle: float, track_pos: float, speed_kmh: float, bot_state: Dict[str, Any]) -> float:
    
    target_pos = 0.0 
    
    steer_gain = max(6.0, 25.0 - (speed_kmh * 0.08))
    target_steer = (angle * steer_gain / PI) - ((track_pos - target_pos) * 0.35)
    
    smooth_factor = 0.4 if speed_kmh < 120 else 0.8
        
    smooth_steer = bot_state.get('prev_steer', 0.0) * smooth_factor + target_steer * (1.0 - smooth_factor)
    steer_val = float(np.clip(smooth_steer, -1.0, 1.0))
    bot_state['prev_steer'] = steer_val
    return steer_val

def calculate_speed_and_pedals(
    speed_kmh: float, 
    angle: float, 
    track_pos: float, 
    steer: float, 
    trk: List[float], 
    wheels: List[float]
) -> Tuple[float, float]:
    
    max_forward_straight = max(trk[8], trk[9], trk[10])
    max_forward_wide = max(trk[6:13]) 

    target_speed = 65.0 + (max_forward_wide * 1.6) 
    target_speed = float(np.clip(target_speed, 65.0, 315.0))
    
    brake_mult = 0.025 + (1.0 - float(np.clip(max_forward_straight / 150.0, 0.0, 1.0))) * 0.22

    steering_intensity = abs(steer)
    is_real_corner = (abs(angle) > 0.06 or steering_intensity > 0.08) and max_forward_straight < 110.0

    if is_real_corner:
        if max_forward_wide < 55.0:
            target_speed = float(np.clip(target_speed, 65.0, 102.0))
            brake_mult = 0.30
        elif max_forward_wide < 95.0: 
            target_speed = float(np.clip(target_speed, 138.0, 182.0))
            brake_mult = 0.10
        elif max_forward_wide < 140.0:
            target_speed = float(np.clip(target_speed, 185.0, 255.0))
            brake_mult = 0.03

    if abs(angle) > 0.12 and max_forward_straight < 50.0:
        target_speed = 70.0
        brake_mult = 0.30

    if abs(track_pos) > 0.98 and speed_kmh > 80:
        target_speed = min(target_speed, 80.0)
        
    accel = 0.0
    brake = 0.0
    
    if speed_kmh < target_speed:
        max_accel = float(np.clip(1.0 - (steering_intensity * 0.85), 0.2, 1.0))
        accel = max_accel
    else:
        raw_brake = (speed_kmh - target_speed) * brake_mult
        max_brake = max(0.1, 1.0 - (steering_intensity * 1.5))
        brake = float(np.clip(min(raw_brake, max_brake), 0.0, 1.0))

    if speed_kmh < 15.0:
        accel = 1.0
        brake = 0.0

    wheel_slip = (wheels[2] + wheels[3]) - (wheels[0] + wheels[1])
    if wheel_slip > 8.0:
        accel = max(0.0, accel - 0.4)
        
    return accel, brake

def calculate_gear(speed_kmh: float, rpm: float, current_gear: int, bot_state: Dict[str, Any]) -> int:
    
    if bot_state['shift_cooldown'] > 0:
        bot_state['shift_cooldown'] -= 1

    gear = current_gear
    if speed_kmh < 25:
        gear = 1
    elif bot_state['shift_cooldown'] == 0:
        if gear < 6 and rpm > 16500:
            gear += 1
            bot_state['shift_cooldown'] = 5
        elif gear > 3 and rpm < 6500:
            gear -= 1
            bot_state['shift_cooldown'] = 5
        elif gear == 3 and rpm < 5000:
            gear -= 1
            bot_state['shift_cooldown'] = 5
        elif gear == 2 and rpm < 4000:
            gear -= 1
            bot_state['shift_cooldown'] = 5
            
    return gear

def drive_and_record(c: Any, session_data: List[Dict[str, Any]], t0: float, start_damage: float, bot_state: Dict[str, Any]) -> bool:
    
    S, R = c.S.d, c.R.d
    if not S:
        raise RaceFinished()

    speed_kmh = S.get('speedX', 0.0)
    angle = S.get('angle', 0.0)
    track_pos = S.get('trackPos', 0.0)
    rpm = S.get('rpm', 0.0)
    current_gear = S.get('gear', 1)
    current_damage = S.get('damage', 0) - start_damage
    
    trk = S.get('track', [100.0] * 19)
    if len(trk) < 19: trk += [100.0] * (19 - len(trk))
    
    wheels = S.get('wheelSpinVel', [0, 0, 0, 0])

    is_on_track = check_bot_status(speed_kmh, track_pos, bot_state)

    cur_lap_time = S.get('lastLapTime', 0.0)
    if cur_lap_time > 0 and cur_lap_time != bot_state.get('last_lap_time', 0.0):
        bot_state['last_lap_time'] = cur_lap_time
        raise LapCompleted(f"Lap completed in {cur_lap_time:.2f}s")

    R['steer'] = calculate_steering(angle, track_pos, speed_kmh, bot_state)
    R['accel'], R['brake'] = calculate_speed_and_pedals(speed_kmh, angle, track_pos, R['steer'], trk, wheels)
    R['gear'] = calculate_gear(speed_kmh, rpm, current_gear, bot_state)

    if is_on_track:
        row_data = {
            'time': time.time() - t0,
            'steer': R['steer'],
            'accel': R['accel'],
            'brake': R['brake'],
            'gear': R['gear'],
            'speedX': speed_kmh,
            'speedY': S.get('speedY', 0.0),
            'speedZ': S.get('speedZ', 0.0),
            'trackPos': track_pos,
            'angle': angle,
            'rpm': rpm,
            'wheelSpinAvg': float(np.mean(wheels)),
            'damage': max(0, current_damage),
        }
        for i in range(19):
            row_data[f'track_{i}'] = trk[i]
        session_data.append(row_data)
        
    return is_on_track

def press_plus(keyboard: Controller):
    print("[BOT] Accelerated speed activated. Focus the TORCS window if necessary.")
    keyboard.press('+')
    keyboard.release('+')

def append_to_master(data: List[Dict[str, Any]]):
    if not data:
        return
    df = pd.DataFrame(data)
    header_flag = not os.path.exists(master_file)
    df.to_csv(master_file, mode='a', header=header_flag, index=False)
    print(f"\n[BOT] Autosave: {len(data)} frames written to disk.")

def main():
    keyboard = Controller()
    C = snakeoil3.Client(p=3001, vision=False)
    session_data: List[Dict[str, Any]] = []
    bot_state = {'shift_cooldown': 0, 'prev_steer': 0.0, 'stuck_frames': 0, 'offtrack_frames': 0}
    last_autosave = 0
    AUTOSAVE_EVERY = 50000

    print("\n[BOT] Data collection bot initialized. Waiting for TORCS...")
    print(f"[BOT] Autosave every {AUTOSAVE_EVERY} frames to prevent data loss.")
    C.get_servers_input()
    press_plus(keyboard)
    t0 = time.time()
    start_damage = C.S.d.get('damage', 0) if C.S.d else 0

    try:
        while True:
            try:
                drive_and_record(C, session_data, t0, start_damage, bot_state)
                C.respond_to_server()
                C.get_servers_input()
                
                # Manual autosave fallback if lap is very long
                if len(session_data) - last_autosave >= AUTOSAVE_EVERY:
                    new_data = session_data[last_autosave:]
                    append_to_master(new_data)
                    last_autosave = len(session_data)
                    
            except RaceFinished:
                print("\n[BOT] Race finished by server.")
                break
            except BotStuck as e:
                print(f"\n[BOT] {e} - Auto-reset! Discarding data from this failed lap...")
                session_data.clear()
                last_autosave = 0
                
                try:
                    C.R.d['meta'] = 1
                    C.respond_to_server()
                    C.shutdown()
                except:
                    pass
                time.sleep(3)
                
                try:
                    C = snakeoil3.Client(p=3001, vision=False)
                    C.get_servers_input()
                    bot_state = {'shift_cooldown': 0, 'prev_steer': 0.0, 'stuck_frames': 0, 'offtrack_frames': 0, 'last_lap_time': 0.0}
                    t0 = time.time()
                    start_damage = C.S.d.get('damage', 0) if C.S.d else 0
                    print("[BOT] Reconnected! Resuming data collection.")
                    press_plus(keyboard)
                except Exception as e2:
                    print(f"[BOT] Reconnection failed: {e2}. Press ENTER after restarting TORCS.")
                    input()
                continue
            except LapCompleted as e:
                print(f"\n[BOT] {e} - Saving data and resetting for the next lap...")
                new_data = session_data[last_autosave:]
                if new_data:
                    append_to_master(new_data)
                
                session_data.clear()
                last_autosave = 0
                
                try:
                    C.R.d['meta'] = 1
                    C.respond_to_server()
                    C.shutdown()
                except:
                    pass
                time.sleep(3)
                
                try:
                    C = snakeoil3.Client(p=3001, vision=False)
                    C.get_servers_input()
                    bot_state = {'shift_cooldown': 0, 'prev_steer': 0.0, 'stuck_frames': 0, 'offtrack_frames': 0, 'last_lap_time': 0.0}
                    t0 = time.time()
                    start_damage = C.S.d.get('damage', 0) if C.S.d else 0
                    print("[BOT] Reconnected for a new perfect lap.")
                    press_plus(keyboard)
                except Exception as e2:
                    print(f"[BOT] Reconnection failed: {e2}. Press ENTER after restarting TORCS.")
                    input()
                continue
            except Exception as e:
                print(f"\n[BOT] Generic error: {e}")
                break

            if len(session_data) % 10 == 0 and session_data:
                last = session_data[-1]
                on_track_str = "OK" if abs(last['trackPos']) < 0.8 else "MARGIN"
                print(f"\r[BOT] Frame: {len(session_data)} | V: {last['speedX']:3.0f} km/h | M: {last['gear']} | Accel: {last['accel']:.2f} | Brake: {last['brake']:.2f} | Pos: {last['trackPos']:+.2f} [{on_track_str}]", end="")

    except KeyboardInterrupt:
        print("\n[BOT] Manual interruption.")
    finally:
        try:
            C.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 1.0, 'gear': 0})
            C.respond_to_server()
            C.shutdown()
        except:
            pass

    print("\n" + "=" * 50)
    print(f"Frames collected in session: {len(session_data)}")
    print("=" * 50)

    remaining = session_data[last_autosave:]
    if len(remaining) >= 100:
        while True:
            choice = input(f"\nDo you want to save the remaining {len(remaining)} frames? [y/n]: ").strip().lower()
            if choice == 'y':
                append_to_master(remaining)
                print("-> SUCCESS!")
                break
            elif choice == 'n':
                print("-> Discarded.")
                break
    elif last_autosave > 0:
        print(f"-> All data already autosaved ({last_autosave} total frames).")
    else:
        print("[BOT] Session too short. Data discarded.")

if __name__ == "__main__":
    main()