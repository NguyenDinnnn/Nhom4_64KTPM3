from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from threading import Lock
import os, pickle, torch, numpy as np, time, uuid
import heapq
from itertools import permutations
from collections import defaultdict
import torch.nn.functional as F
import copy

from app.robot_env import GridWorldEnv
from clients.train_a2c import ActorCritic

# ---------------------------
# App setup
# ---------------------------
app = FastAPI(title="RL Robot API - Session Based", version="2.0.0")
app.mount("/web", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../clients/web")), name="web")

# ---------------------------
# Session Management
# ---------------------------
sessions: Dict[str, 'Session'] = {}
_sessions_lock = Lock()

# ---------------------------
# Models dir
# ---------------------------
models_dir = os.path.join(os.path.dirname(__file__), "../clients/models")
os.makedirs(models_dir, exist_ok=True)

# ---------------------------
# Load base Q-tables and models once at startup
# ---------------------------
def load_q_table(file_path):
    q_table = defaultdict(lambda: {a: 0.0 for a in ['up', 'right', 'down', 'left']})
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
            q_table.update(loaded_data)
    return q_table

base_mc_Q = load_q_table(os.path.join(models_dir, "mc_qtable.pkl"))
base_ql_Q = load_q_table(os.path.join(models_dir, "qlearning_qtable.pkl"))

# Load A2C model
base_a2c_model = ActorCritic(in_channels=5, height=8, width=10, n_actions=4)
a2c_model_file = os.path.join(models_dir, "a2c_model.pth")
if os.path.exists(a2c_model_file):
    try:
        base_a2c_model.load_state_dict(torch.load(a2c_model_file))
        print("✅ Base A2C model loaded successfully")
    except RuntimeError:
        print("⚠️ Could not load A2C checkpoint. Using a new model.")
base_a2c_model.eval()

# ---------------------------
# Session Class to hold state for each user
# ---------------------------
class Session:
    def __init__(self):
        self.env = GridWorldEnv(
            width=10, height=8,
            start=(0,0), goal=(9,7),
            obstacles=[(1,1),(2,3),(4,4),(5,1),(7,6)],
            waypoints=[(3,2),(6,5)],
            max_steps=500
        )
        # Deep copy base models/tables for this session
        self.mc_Q = copy.deepcopy(base_mc_Q)
        self.ql_Q = copy.deepcopy(base_ql_Q)
        self.a2c_model = copy.deepcopy(base_a2c_model)
        self.epsilon = 1.0
        self.actions = ['up', 'right', 'down', 'left']
        self.alpha = 0.1
        self.gamma = 0.99
        self.lock = Lock() # Lock for each session's operations

# ---------------------------
# Request Models (with session_id)
# ---------------------------
class SessionRequest(BaseModel):
    session_id: str

class ResetRequest(SessionRequest):
    width: Optional[int]=None
    height: Optional[int]=None
    start: Optional[Tuple[int,int]]=None
    goal: Optional[Tuple[int,int]]=None
    waypoints: Optional[List[Tuple[int,int]]]=None
    obstacles: Optional[List[Tuple[int,int]]]=None
    max_steps: Optional[int]=None

class ActionInput(SessionRequest):
    action: Optional[int]=None
    action_name: Optional[str]=None

class AlgorithmRequest(SessionRequest):
    algorithm: str

class AStarRequest(SessionRequest):
    goal: Optional[Tuple[int,int]] = None

# ---------------------------
# A* functions (unchanged)
# ---------------------------
def a_star(start, goal, obstacles, width, height):
    open_set = []
    heapq.heappush(open_set, (0+abs(start[0]-goal[0])+abs(start[1]-goal[1]), 0, start, [start]))
    visited = set()
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal: return path
        if current in visited: continue
        visited.add(current)
        x,y = current
        for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<width and 0<=ny<height and (nx,ny) not in obstacles:
                heapq.heappush(open_set, (g+1+abs(nx-goal[0])+abs(ny-goal[1]), g+1, (nx,ny), path+[(nx,ny)]))
    return []

def plan_path_through_waypoints(start, waypoints, goal, obstacles, width, height):
    best_path, min_len = None, float('inf')
    for order in permutations(waypoints):
        path, curr, valid = [], start, True
        for wp in order:
            sub_path = a_star(curr, wp, obstacles, width, height)
            if not sub_path: valid = False; break
            path += sub_path[:-1]; curr = wp
        if not valid: continue
        sub_path = a_star(curr, goal, obstacles, width, height)
        if not sub_path: continue
        path += sub_path
        if len(path) < min_len: min_len, best_path = len(path), path
    return best_path or []

# ---------------------------
# Extend GridWorldEnv for A* step (RL-style reward)
# ---------------------------
def step_to_rl(self, target):
    self.state = target
    self.steps += 1
    reward = -0.1
    done = False
    if target in self.waypoints and target not in self.visited_waypoints:
        self.visited_waypoints.add(target)
        reward = 1
    if target == self.goal and len(self.visited_waypoints) == len(self.waypoints):
        done = True
        reward = 10
    info = {"note": "Auto move by A* (RL reward)"}
    return target, reward, done, info
GridWorldEnv.step_to = step_to_rl

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/sessions/create")
def create_session():
    """Creates a new simulation session."""
    session_id = str(uuid.uuid4())
    with _sessions_lock:
        sessions[session_id] = Session()
    return {"session_id": session_id}

@app.post("/map")
def get_map(req: SessionRequest):
    session = sessions.get(req.session_id)
    if not session: return {"error": "Invalid session ID"}
    with session.lock:
        return {"map": session.env.get_map()}

@app.post("/reset")
def reset(req: ResetRequest):
    session = sessions.get(req.session_id)
    if not session: return {"error": "Invalid session ID"}
    with session.lock:
        w = req.width or session.env.width
        h = req.height or session.env.height
        s = req.start or session.env.start
        g = req.goal or session.env.goal
        wp = req.waypoints if req.waypoints is not None else list(session.env.waypoints)
        ob = req.obstacles if req.obstacles is not None else list(session.env.obstacles)
        ms = req.max_steps if req.max_steps is not None else 100

        session.env = GridWorldEnv(w, h, s, g, ob, wp, max_steps=ms)
        state = session.env.reset(max_steps=ms)
        # Reset epsilon for this session
        session.epsilon = 1.0
        return {"state": state, "map": session.env.get_map(), "ascii": session.env.render_ascii()}

@app.post("/state")
def get_state(req: SessionRequest):
    session = sessions.get(req.session_id)
    if not session: return {"error": "Invalid session ID"}
    with session.lock:
        return {
            "state": session.env.get_state(),
            "steps": session.env.steps,
            "visited_waypoints": list(session.env.visited_waypoints),
            "ascii": session.env.render_ascii()
        }

@app.post("/step")
def step(inp: ActionInput):
    session = sessions.get(inp.session_id)
    if not session: return {"error": "Invalid session ID"}
    with session.lock:
        try:
            if inp.action_name is not None:
                s,r,done,info = session.env.step_by_name(inp.action_name)
            elif inp.action is not None:
                s,r,done,info = session.env.step(inp.action)
            else:
                return {"error": "No action provided"}
            return {
                "state": s, "reward": r, "done": done, "info": info,
                "steps": session.env.steps,
                "visited_waypoints": list(session.env.visited_waypoints),
                "ascii": session.env.render_ascii()
            }
        except ValueError as e:
            return {"error": str(e)}

@app.post("/step_algorithm")
def step_algorithm(req: AlgorithmRequest):
    session = sessions.get(req.session_id)
    if not session: return {"error": "Invalid session ID"}
    
    algo = req.algorithm
    with session.lock:
        state_xy = session.env.get_state()
        done = False
        reward = 0

        def encode_visited(wp_list, visited_set):
            return sum(1 << i for i, wp in enumerate(wp_list) if wp in visited_set)
        
        visited_code = encode_visited(session.env.waypoints, session.env.visited_waypoints)
        full_state = (state_xy[0], state_xy[1], visited_code)
        
        action_name = ""

        # Sửa tên 'MC' thành 'TD-Update' để phản ánh đúng bản chất
        # Client sẽ gửi 'MC', server xử lý như 'TD-Update'
        if algo == "MC": 
            algo = "TD-Update"

        if algo == "TD-Update":
            if np.random.rand() > session.epsilon:
                if full_state in session.mc_Q and any(session.mc_Q[full_state].values()):
                     action_name = max(session.mc_Q[full_state], key=session.mc_Q[full_state].get)
                else:
                     action_name = np.random.choice(session.actions)
            else:
                action_name = np.random.choice(session.actions)
            
            action_idx = session.actions.index(action_name)
            next_state_xy, r, done, _ = session.env.step(action_idx)
            
            # **FIX**: Lấy next_state_full đúng
            next_visited_code = encode_visited(session.env.waypoints, session.env.visited_waypoints)
            next_state_full = (next_state_xy[0], next_state_xy[1], next_visited_code)
            
            # Đây là công thức của Q-learning (một dạng TD), không phải MC thuần túy
            G = r + session.gamma * max(session.mc_Q[next_state_full].values())
            session.mc_Q[full_state][action_name] += session.alpha * (G - session.mc_Q[full_state][action_name])
            
            reward = r
            state_xy = next_state_xy
        
        elif algo == "Q-learning":
            if np.random.rand() < session.epsilon:
                action_name = np.random.choice(session.actions)
            else:
                action_name = max(session.ql_Q[full_state], key=session.ql_Q[full_state].get)
            
            action_idx = session.actions.index(action_name)
            next_state_xy, r, done, _ = session.env.step(action_idx)
            
            next_visited_code = encode_visited(session.env.waypoints, session.env.visited_waypoints)
            next_state_full = (next_state_xy[0], next_state_xy[1], next_visited_code)
            
            session.ql_Q[full_state][action_name] += session.alpha * (
                r + session.gamma * max(session.ql_Q[next_state_full].values()) - session.ql_Q[full_state][action_name]
            )
            
            state_xy = next_state_xy
            reward = r
            session.epsilon = max(0.1, session.epsilon * 0.995)

        elif algo == "A2C":
            state_tensor = session.env.build_grid_state().unsqueeze(0)
            session.a2c_model.eval()
            with torch.no_grad():
                policy_logits, _ = session.a2c_model(state_tensor)
                action_probs = F.softmax(policy_logits, dim=-1).squeeze(0)
                action_idx = torch.multinomial(action_probs, 1).item()
            
            next_state_xy, r, done, _ = session.env.step(action_idx)
            state_xy = next_state_xy
            reward = r

        return {
            "state": state_xy,
            "reward": reward,
            "done": done or session.env.steps >= session.env.max_steps,
            "steps": session.env.steps,
            "visited_waypoints": list(session.env.visited_waypoints)
        }

@app.post("/run_a_star")
def run_a_star(req: AStarRequest):
    session = sessions.get(req.session_id)
    if not session: return {"error": "Invalid session ID"}
    with session.lock:
        start_time = time.time()
        
        start_pos = session.env.get_state()
        path = plan_path_through_waypoints(start_pos, session.env.waypoints, req.goal or session.env.goal,
                                           session.env.obstacles, session.env.width, session.env.height)
        if not path:
            return {"error": "Không tìm thấy đường đi qua tất cả waypoint"}
        
        # Reset env for this run
        session.env.reset()
        total_reward = 0
        rewards_over_time = []
        for node in path[1:]:
            s, r, done, info = session.env.step_to(node)
            total_reward += r
            rewards_over_time.append(total_reward)

        done = (session.env.state == session.env.goal and len(session.env.visited_waypoints) == len(session.env.waypoints))
        elapsed_time = time.time() - start_time
        
        return {
            "algorithm": "A*", "path": path, "state": session.env.get_state(),
            "reward": total_reward, "done": done, "steps": session.env.steps,
            "visited_waypoints": list(session.env.visited_waypoints), "info": {},
            "ascii": session.env.render_ascii(), "elapsed_time": elapsed_time,
            "rewards_over_time": rewards_over_time
        }

# ---------------------------
# Save Endpoints (Lưu vào file chung)
# ---------------------------
@app.post("/save_qlearning")
def save_qlearning(req: SessionRequest):
    session = sessions.get(req.session_id)
    if not session: return {"error": "Invalid session ID"}
    # Ghi đè file chung với dữ liệu từ session này
    with open(os.path.join(models_dir, 'qlearning_qtable.pkl'), 'wb') as f:
        pickle.dump(dict(session.ql_Q), f)
    return {"status": "Q-learning Q-table saved from session"}

@app.post("/save_mc")
def save_mc(req: SessionRequest):
    session = sessions.get(req.session_id)
    if not session: return {"error": "Invalid session ID"}
    with open(os.path.join(models_dir, 'mc_qtable.pkl'), 'wb') as f:
        pickle.dump(dict(session.mc_Q), f)
    return {"status": "MC Q-table saved from session"}

@app.post("/save_a2c")
def save_a2c(req: SessionRequest):
    session = sessions.get(req.session_id)
    if not session: return {"error": "Invalid session ID"}
    torch.save(session.a2c_model.state_dict(), os.path.join(models_dir, 'a2c_model.pth'))
    return {"status": "A2C model saved from session"}

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web/index.html")