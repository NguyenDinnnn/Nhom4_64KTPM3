from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional
from threading import Lock
import os, pickle, torch, numpy as np
import heapq
from itertools import permutations

from app.robot_env import GridWorldEnv
from clients.train_mc import Q as mc_Q
from clients.train_qlearning import Q as ql_Q
from clients.train_a2c import ActorCritic

# ---------------------------
# App setup
# ---------------------------
app = FastAPI(title="RL Robot API", version="1.0.0")
_env_lock = Lock()
app.mount("/web", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../clients/web")), name="web")

# ---------------------------
# Environment
# ---------------------------
width, height = 10, 8
start = (0,0)
goal = (9,7)
waypoints = [(3,2),(6,5)]
obstacles = [(1,1),(2,3),(4,4),(5,1),(7,6)]
env = GridWorldEnv(width=width, height=height, start=start, goal=goal, obstacles=obstacles, waypoints=waypoints)

# ---------------------------
# Models dir
# ---------------------------
models_dir = os.path.join(os.path.dirname(__file__), "../clients/models")
os.makedirs(models_dir, exist_ok=True)

# ---------------------------
# Load MC
# ---------------------------
mc_qfile = os.path.join(models_dir, "mc_qtable.pkl")
if os.path.exists(mc_qfile):
    with open(mc_qfile, "rb") as f:
        mc_Q = pickle.load(f)
else:
    mc_Q = {}

# ---------------------------
# Load Q-learning
# ---------------------------
ql_qfile = os.path.join(models_dir, "qlearning_qtable.pkl")
if os.path.exists(ql_qfile):
    with open(ql_qfile, "rb") as f:
        ql_Q = pickle.load(f)
else:
    ql_Q = {}

# ---------------------------
# Load A2C
# ---------------------------
a2c_model_file = os.path.join(models_dir, "a2c_model.pth")
in_channels = 5
height, width = env.height, env.width
n_actions = len(env.ACTIONS)
a2c_model = ActorCritic(in_channels=in_channels, height=height, width=width, n_actions=n_actions)
if os.path.exists(a2c_model_file):
    try:
        a2c_model.load_state_dict(torch.load(a2c_model_file))
        a2c_model.eval()
        print("✅ A2C model loaded successfully")
    except RuntimeError:
        print("⚠️ Không load được A2C checkpoint. Sẽ dùng model mới.")

# ---------------------------
# Action list & RL params
# ---------------------------
actions = ['up', 'right', 'down', 'left']
alpha = 0.1
gamma = 0.99
epsilon = 1.0

# ---------------------------
# Request Models
# ---------------------------
class ResetRequest(BaseModel):
    width: Optional[int]=None
    height: Optional[int]=None
    start: Optional[Tuple[int,int]]=None
    goal: Optional[Tuple[int,int]]=None
    waypoints: Optional[List[Tuple[int,int]]]=None
    obstacles: Optional[List[Tuple[int,int]]]=None
    max_steps: Optional[int]=None

class ActionInput(BaseModel):
    action: Optional[int]=None
    action_name: Optional[str]=None

class AlgorithmRequest(BaseModel):
    algorithm: str

class AStarRequest(BaseModel):
    goal: Optional[Tuple[int,int]] = None

# ---------------------------
# A* Search function
# ---------------------------
def a_star(start, goal, obstacles, width, height):
    open_set = []
    heapq.heappush(open_set, (0+abs(start[0]-goal[0])+abs(start[1]-goal[1]), 0, start, [start]))
    visited = set()
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        x,y = current
        for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<width and 0<=ny<height and (nx,ny) not in obstacles:
                heapq.heappush(open_set, (g+1+abs(nx-goal[0])+abs(ny-goal[1]), g+1, (nx,ny), path+[(nx,ny)]))
    return []

def plan_path_through_waypoints(start, waypoints, goal, obstacles, width, height):
    best_path = None
    min_len = float('inf')
    for order in permutations(waypoints):
        path = []
        curr = start
        valid = True
        for wp in order:
            sub_path = a_star(curr, wp, obstacles, width, height)
            if not sub_path:
                valid = False
                break
            path += sub_path[:-1]
            curr = wp
        if not valid:
            continue
        sub_path = a_star(curr, goal, obstacles, width, height)
        if not sub_path:
            continue
        path += sub_path
        if len(path) < min_len:
            min_len = len(path)
            best_path = path
    return best_path or []

# ---------------------------
# Extend GridWorldEnv for A* step (RL-style reward)
# ---------------------------
def step_to_rl(self, target):
    self.state = target
    self.steps += 1
    reward = -0.1  # bước di chuyển bình thường
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
# Endpoints
# ---------------------------
@app.get("/map")
def get_map():
    with _env_lock:
        return {"map": env.get_map()}

@app.post("/reset")
def reset(req: ResetRequest):
    global env
    with _env_lock:
        w = req.width or env.width
        h = req.height or env.height
        s = req.start or env.start
        g = req.goal or env.goal
        wp = req.waypoints if req.waypoints is not None else list(env.waypoints)
        ob = req.obstacles if req.obstacles is not None else list(env.obstacles)
        ms = req.max_steps if req.max_steps is not None else env.max_steps

        env = GridWorldEnv(width=w, height=h, start=s, goal=g, obstacles=ob, waypoints=wp, max_steps=ms)
        state = env.reset(max_steps=ms)
        return {"state": state, "map": env.get_map(), "ascii": env.render_ascii()}

@app.get("/state")
def get_state():
    with _env_lock:
        return {
            "state": env.get_state(),
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "ascii": env.render_ascii()
        }

@app.post("/step")
def step(inp: ActionInput):
    with _env_lock:
        try:
            if inp.action_name is not None:
                s,r,done,info = env.step_by_name(inp.action_name)
            elif inp.action is not None:
                s,r,done,info = env.step(inp.action)
            else:
                return {"error":"No action provided"}
            return {
                "state": s,
                "reward": r,
                "done": done,
                "info": info,
                "steps": env.steps,
                "visited_waypoints": list(env.visited_waypoints),
                "ascii": env.render_ascii()
            }
        except ValueError as e:
            return {"error": str(e)}

# ---------------------------
# Run RL Algorithm
# ---------------------------
@app.post("/run_algorithm")
def run_algorithm(req: AlgorithmRequest):
    global epsilon
    algo = req.algorithm
    with _env_lock:
        state_xy = env.get_state()
        done = False
        reward = 0

        def encode_visited(wp_list, visited_set):
            code = 0
            for i, wp in enumerate(wp_list):
                if wp in visited_set:
                    code |= (1 << i)
            return code

        visited_code = encode_visited(env.waypoints, env.visited_waypoints)

        if algo == "MC":
            full_state = (state_xy[0], state_xy[1], visited_code)
            if full_state in mc_Q and np.random.rand() > epsilon:
                action = max(mc_Q[full_state], key=mc_Q[full_state].get)
            else:
                action = np.random.choice(actions)
            action_idx = actions.index(action)
            next_state, reward, done, _ = env.step(action_idx)

        elif algo == "A2C":
            state_tensor = env.build_grid_state().unsqueeze(0)
            a2c_model.eval()
            with torch.no_grad():
                policy_logits, _ = a2c_model(state_tensor)
                action_probs = torch.softmax(policy_logits, dim=-1)

                remaining_waypoints = [wp for wp in env.waypoints if wp not in env.visited_waypoints]
                preferred_actions = []
                x, y = env.state
                for idx, (dx, dy) in enumerate(env.ACTIONS):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < env.width and 0 <= ny < env.height and (nx, ny) not in env.obstacles:
                        if (nx, ny) in remaining_waypoints:
                            preferred_actions.append(idx)

                if preferred_actions:
                    preferred_probs = action_probs[0, preferred_actions]
                    preferred_probs /= preferred_probs.sum()
                    action_idx = torch.multinomial(preferred_probs, 1).item()
                    action_idx = preferred_actions[action_idx]
                else:
                    action_idx = torch.multinomial(action_probs, 1).item()

            next_state, reward, done, _ = env.step(action_idx)

            try:
                tmp = a2c_model_file + ".tmp"
                torch.save(a2c_model.state_dict(), tmp)
                os.replace(tmp, a2c_model_file)
            except Exception as e:
                print("Warning: failed to save A2C model:", repr(e))

        elif algo=="Q-learning":
            state_tuple = (state_xy[0], state_xy[1], visited_code)
            if state_tuple not in ql_Q:
                ql_Q[state_tuple] = {a: 0.0 for a in actions}

            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                action = max(ql_Q[state_tuple], key=ql_Q[state_tuple].get)

            action_idx = actions.index(action)
            next_state, reward, done, _ = env.step(action_idx)

            next_visited_code = encode_visited(env.waypoints, env.visited_waypoints)
            next_state_tuple = (next_state[0], next_state[1], next_visited_code)
            if next_state_tuple not in ql_Q:
                ql_Q[next_state_tuple] = {a: 0.0 for a in actions}

            ql_Q[state_tuple][action] += alpha * (
                reward + gamma * max(ql_Q[next_state_tuple].values()) - ql_Q[state_tuple][action]
            )
            epsilon = max(0.1, epsilon * 0.995)

        else:
            return {"error": "Thuật toán không hợp lệ"}

        return {
            "algorithm": algo,
            "state": next_state,
            "reward": reward,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "done": done
        }

# ---------------------------
# Run A* Algorithm (RL-style reward + waypoint bắt buộc)
# ---------------------------
@app.post("/run_a_star")
def run_a_star(req: AStarRequest):
    with _env_lock:
        start = env.get_state()
        path = plan_path_through_waypoints(start, env.waypoints, req.goal or env.goal, env.obstacles, env.width, env.height)
        if not path:
            return {"error": "Không tìm thấy đường đi qua tất cả waypoint"}

        total_reward = 0
        info_list = []
        for node in path[1:]:
            s, r, done, info = env.step_to(node)
            total_reward += r
            info_list.append(info)
        done = (env.state == env.goal and len(env.visited_waypoints) == len(env.waypoints))
        return {
            "algorithm": "A*",
            "path": path,
            "state": env.get_state(),
            "reward": total_reward,
            "done": done,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "info": info_list,
            "ascii": env.render_ascii()
        }

# ---------------------------
# Auto Run Endpoint
# ---------------------------
@app.post("/auto_run")
def auto_run():
    with _env_lock:
        start = env.get_state()
        # Tìm đường đi từ start -> waypoint -> goal
        path = plan_path_through_waypoints(start, env.waypoints, env.goal,
                                           env.obstacles, env.width, env.height)
        if not path:
            return {"error": "Không tìm thấy đường đi qua tất cả waypoint"}

        # Reset lại env để chạy từ đầu
        env.reset()
        full_path = [env.state]
        total_reward = 0
        info_list = []

        for node in path[1:]:
            s, r, done, info = env.step_to(node)
            total_reward += r
            full_path.append(s)
            info_list.append(info)

        done = (env.state == env.goal and len(env.visited_waypoints) == len(env.waypoints))

        return {
            "algorithm": "auto_run (A*)",
            "path": full_path,
            "state": env.get_state(),
            "reward": total_reward,
            "done": done,
            "steps": env.steps,
            "visited_waypoints": list(env.visited_waypoints),
            "info": info_list,
            "ascii": env.render_ascii()
        }
    
# ---------------------------
# Save Endpoints
# ---------------------------
@app.post("/save_qlearning")
def save_qlearning():
    with open(os.path.join(models_dir, 'qlearning_qtable.pkl'), 'wb') as f:
        pickle.dump(ql_Q, f)
    return {"status": "Q-learning Q-table saved"}

@app.post("/save_mc")
def save_mc():
    with open(os.path.join(models_dir, 'mc_qtable.pkl'), 'wb') as f:
        pickle.dump(mc_Q, f)
    return {"status": "MC Q-table saved"}

@app.post("/save_a2c")
def save_a2c():
    torch.save(a2c_model.state_dict(), os.path.join(models_dir, 'a2c_model.pth'))
    return {"status": "A2C model saved"}

# ---------------------------
# Root
# ---------------------------
@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
