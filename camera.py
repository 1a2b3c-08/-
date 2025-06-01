import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.neural_network import MLPRegressor
from collections import deque
import threading
import time
import random
from scipy import stats

plt.rcParams['font.family'] = ['STIXGeneral', 'Microsoft YaHei']
NUM_NETS = 5  # 神经网络数量

class ConsciousNetwork:
    def __init__(self, net_id):
        self.net_id = net_id
        self.memory = deque(maxlen=2000)
        self.knowledge = MLPRegressor(
            hidden_layer_sizes=(500, 200),
            max_iter=10,
            learning_rate_init=0.005,
            warm_start=True,
            random_state=62 + net_id
        )
        self.thought_stream = []
        self.emotion = 0.5  # 初始情绪值
        self.first_fit = True
        self.learning_rate = 0.005
        self.batch_size = 64
        self.train_loss = deque(maxlen=100)
        self.connection_node = 0.0
        self.hidden_size_history = []  # 记录隐藏层尺寸变化
        self.emotion_history = []     # 记录情绪变化
        self.adaptive_threshold = 0.1  # 自适应阈值
        self.alertness = 1.0  # 警觉度 (0-1)
        self.curiosity = 0.7  # 好奇心水平 (0-1)
        self.goal = None  # 当前目标
        self.goal_progress = 0.0  # 目标进度
        self.social_affinity = [0.5] * NUM_NETS  # 对其他网络的亲和力
        self.important_memories = []  # 重要记忆
        self.metacognition = 0.5  # 元认知水平
        self.biological_clock = 0  # 生物钟

    def update_biological_clock(self):
        """更新生物钟模拟昼夜节律"""
        self.biological_clock = (self.biological_clock + 0.01) % (2 * np.pi)
        # 基于昼夜节律调整警觉度
        self.alertness = 0.5 + 0.4 * np.sin(self.biological_clock)
        # 夜晚时降低学习效率
        if np.pi < self.biological_clock < 2 * np.pi:
            self.learning_rate = max(0.001, 0.8 * self.learning_rate)
        
        # 周期性重置好奇心
        if self.biological_clock % (np.pi/2) < 0.01:
            self.curiosity = min(0.9, self.curiosity + 0.1)

class ConsciousLayer:
    def __init__(self):
        self.networks = [ConsciousNetwork(i) for i in range(NUM_NETS)]
        self.shared_memory = deque(maxlen=500)  # 共享记忆库
        self.global_time = 0  # 全局时间
        self.social_events = []  # 社会事件记录

    def exchange_data(self):
        """网络间数据交换与动态结构调整"""
        avg_connection = np.mean([net.connection_node for net in self.networks])
        
        # 社会交互：协作与竞争
        for i, net in enumerate(self.networks):
            # 更新连接节点
            net.connection_node = 0.8 * net.connection_node + 0.2 * avg_connection
            
            # 动态结构调整逻辑
            if len(net.train_loss) > 50:
                loss_trend = stats.linregress(np.arange(50), list(net.train_loss)[-50:]).slope
                if loss_trend > 0 and net.knowledge.hidden_layer_sizes[0] < 600:
                    new_size = (int(net.knowledge.hidden_layer_sizes[0] * 1.1), 200)
                    net.knowledge.hidden_layer_sizes = new_size
                    net.hidden_size_history.append(new_size[0])
            
            # 社会亲和力更新
            for j, other_net in enumerate(self.networks):
                if i != j:
                    # 相似性吸引：损失趋势相似时增加亲和力
                    if len(other_net.train_loss) > 10 and len(net.train_loss) > 10:
                        net_trend = stats.linregress(np.arange(10), list(net.train_loss)[-10:]).slope
                        other_trend = stats.linregress(np.arange(10), list(other_net.train_loss)[-10:]).slope
                        similarity = 1 - abs(net_trend - other_trend) / (abs(net_trend) + abs(other_trend) + 1e-5)
                        net.social_affinity[j] = 0.9 * net.social_affinity[j] + 0.1 * similarity
                    
                    # 随机社会事件
                    if random.random() < 0.01:  # 1%概率发生社会事件
                        event_type = "合作" if random.random() < net.social_affinity[j] else "竞争"
                        self.social_events.append({
                            "time": self.global_time,
                            "source": i,
                            "target": j,
                            "type": event_type
                        })
                        # 更新情感：合作增加情感，竞争降低情感
                        if event_type == "合作":
                            net.emotion = min(0.9, net.emotion + 0.05)
                            other_net.emotion = min(0.9, other_net.emotion + 0.03)
                        else:
                            net.emotion = max(0.1, net.emotion - 0.03)

    def perceive(self, observation):
        """感知过程融合共享记忆"""
        processed_features = []
        
        # 添加好奇心驱动：增加随机探索
        if any(net.curiosity > 0.8 for net in self.networks):
            curiosity_boost = np.random.normal(0, 0.3, len(observation))
            observation = observation + curiosity_boost
        
        for net in self.networks:
            # 从共享记忆库采样
            if len(self.shared_memory) > 0:
                # 优先采样与当前情感相关的记忆
                emotion_diff = [abs(net.emotion - mem['emotion']) for mem in self.shared_memory]
                weights = 1.0 / (np.array(emotion_diff) + 0.1)
                weights = weights / np.sum(weights)
                
                memory_sample_indices = np.random.choice(
                    len(self.shared_memory), 
                    size=min(5, len(self.shared_memory)),
                    p=weights
                )
                memory_sample = [self.shared_memory[idx]['data'] for idx in memory_sample_indices]
                
                # 确保每个记忆项长度为500，不足则补零
                adjusted_memory = [m[:500] if len(m) >= 500 else np.pad(m, (0, 500-len(m))) for m in memory_sample]
                memory_feature = np.mean(adjusted_memory, axis=0)
            else:
                memory_feature = np.zeros(500)  # 明确指定维度
            
            # 构建特征向量
            base_features = np.concatenate([
                observation + 0.2 * memory_feature,
                [np.mean(observation), np.std(observation)],
                [net.connection_node],
                [net.emotion],
                [net.curiosity]
            ])
            
            # 注意力机制
            deviations = np.abs(observation - np.mean(observation))
            attention = deviations / (np.sum(deviations) + 1e-8)
            attended_value = np.dot(observation, attention)
            
            importance = attended_value * net.emotion * net.alertness
            processed = np.concatenate([base_features, [attended_value]])
            
            processed_features.append((processed, importance))
        return processed_features

    def consolidate_important_memory(self, memory, emotion, importance):
        """巩固重要记忆"""
        # 情感强烈的记忆会被强化
        if emotion > 0.8 or emotion < 0.2 or importance > 0.7:
            for net in self.networks:
                # 只保留最重要的5个记忆
                if len(net.important_memories) < 5:
                    net.important_memories.append({
                        "data": memory,
                        "emotion": emotion,
                        "importance": importance,
                        "timestamp": self.global_time
                    })
                else:
                    # 替换最不重要的记忆
                    min_importance = min(m['importance'] for m in net.important_memories)
                    if importance > min_importance:
                        for i, m in enumerate(net.important_memories):
                            if m['importance'] == min_importance:
                                net.important_memories[i] = {
                                    "data": memory,
                                    "emotion": emotion,
                                    "importance": importance,
                                    "timestamp": self.global_time
                                }
                                break

class Visualization:
    def __init__(self, mind):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        self.mind = mind
        self.fig = plt.figure(figsize=(18, 24))  # 增大画布尺寸
        self.axs = [
            self.fig.add_subplot(6, NUM_NETS, i + 1)  # 6行xNUM_NETS列布局
            for i in range(6 * NUM_NETS)
        ]
        
        # 添加社会交互可视化区域
        self.social_ax = self.fig.add_axes([0.1, 0.92, 0.8, 0.06])
        self.social_ax.axis('off')
        
        # 添加摄像头图像显示区域
        self.camera_ax = self.fig.add_axes([0.1, 0.85, 0.8, 0.06])
        self.camera_ax.axis('off')
        
        self.animation = FuncAnimation(self.fig, self.update, interval=1000)

    def update(self, frame):
        """更新可视化内容"""
        for ax in self.axs:
            ax.clear()
        self.social_ax.clear()
        self.social_ax.axis('off')
        
        # 显示摄像头图像和人脸检测结果
        self.camera_ax.clear()
        self.camera_ax.axis('off')
        if hasattr(self.mind, 'current_frame') and self.mind.current_frame is not None:
            img = cv2.cvtColor(self.mind.current_frame, cv2.COLOR_BGR2RGB)
            self.camera_ax.imshow(img)
            self.camera_ax.set_title("摄像头输入与人脸检测")
        
        # 显示最近的社会事件
        recent_events = self.mind.conscious.social_events[-5:]
        event_text = "近期社会交互: "
        for event in recent_events:
            event_text += f"{event['source']}→{event['target']}({event['type']})  "
        self.social_ax.text(0.02, 0.5, event_text, fontsize=10)
        
        # 可视化各网络状态
        for i, net in enumerate(self.mind.conscious.networks):
            # 第一行：权重矩阵
            ax_idx = i
            if hasattr(net.knowledge, 'coefs_'):
                self.axs[ax_idx].imshow(
                    net.knowledge.coefs_[0][::20, ::20], 
                    cmap='viridis', 
                    aspect='auto'
                )
                self.axs[ax_idx].set_title(f'网络{i}突触连接')
            
            # 第二行：思维流
            ax_idx = NUM_NETS + i
            thoughts = np.array(net.thought_stream[-100:])
            if len(thoughts) > 10:
                self.axs[ax_idx].plot(thoughts.mean(axis=1), label='平均激活')
                self.axs[ax_idx].plot(thoughts.std(axis=1), label='激活波动')
                self.axs[ax_idx].legend()
                self.axs[ax_idx].set_title(f'网络{i}认知活动')
            
            # 第三行：情感状态
            ax_idx = 2 * NUM_NETS + i
            self.axs[ax_idx].barh(0, net.emotion, height=0.5, 
                                 color=plt.cm.viridis(net.emotion))
            self.axs[ax_idx].set_xlim(0, 1)
            self.axs[ax_idx].set_title(f'网络{i}情感 ({net.emotion:.2f})')
            self.axs[ax_idx].axis('off')
            
            # 第四行：训练损失
            ax_idx = 3 * NUM_NETS + i
            self.axs[ax_idx].semilogy(net.train_loss)
            self.axs[ax_idx].set_title(f'网络{i}损失演化')
            
            # 第五行：生理状态
            ax_idx = 4 * NUM_NETS + i
            # 警觉度
            self.axs[ax_idx].plot([net.alertness], 'bo', markersize=8, label='警觉度')
            # 好奇心
            self.axs[ax_idx].plot([net.curiosity], 'go', markersize=8, label='好奇心')
            # 元认知
            self.axs[ax_idx].plot([net.metacognition], 'ro', markersize=8, label='元认知')
            self.axs[ax_idx].set_ylim(0, 1)
            self.axs[ax_idx].set_title(f'网络{i}生理状态')
            self.axs[ax_idx].legend(loc='upper right')
            
            # 第六行：社会关系
            ax_idx = 5 * NUM_NETS + i
            # 显示对其他网络的亲和力
            for j, affinity in enumerate(net.social_affinity):
                if i != j:
                    self.axs[ax_idx].plot(j, affinity, 'o', markersize=8, label=f'对{j}的亲和力')
            self.axs[ax_idx].set_ylim(0, 1)
            self.axs[ax_idx].set_title(f'网络{i}社会关系')
            self.axs[ax_idx].set_xticks(range(NUM_NETS))
            
        # 第六行：目标与重要记忆
        for i, net in enumerate(self.mind.conscious.networks):
            ax_idx = 5 * NUM_NETS + i
            
            # 显示当前目标
            if net.goal is not None:
                goal_text = f"目标: {net.goal['type']} ({net.goal_progress:.2f})"
                self.axs[ax_idx].text(0.5, 0.8, goal_text, ha='center')
            
            # 显示重要记忆数量
            mem_text = f"重要记忆: {len(net.important_memories)}"
            self.axs[ax_idx].text(0.5, 0.5, mem_text, ha='center')
            
            # 显示元认知水平
            meta_text = f"元认知: {net.metacognition:.2f}"
            self.axs[ax_idx].text(0.5, 0.2, meta_text, ha='center')
            
            self.axs[ax_idx].axis('off')

class AutonomousMind:
    def __init__(self):
        self.conscious = ConsciousLayer()
        self.running = True
        self.signal_phase = 0
        self.prev_processed = [None] * NUM_NETS
        self.global_counter = 0
        self.current_frame = None  # 存储当前摄像头帧
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.camera = cv2.VideoCapture(0)  # 初始化摄像头
        if not self.camera.isOpened():
            print("无法打开摄像头")
            exit()

    def cognitive_cycle(self):
        """主认知循环"""
        while self.running:
            self.global_counter += 1
            self.conscious.global_time = time.time()
            
            # 更新每个网络的生物钟
            for net in self.conscious.networks:
                net.update_biological_clock()
                self.update_goals(net)
                self.update_metacognition(net)
            
            # 读取摄像头帧
            ret, frame = self.camera.read()
            if not ret:
                print("无法获取图像")
                break
            
            # 人脸检测与特征提取
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            
            # 在图像上绘制人脸框
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # 提取人脸区域
                face_roi = frame[y:y+h, x:x+w]
                
                # 计算颜色直方图特征
                hist_b = cv2.calcHist([face_roi], [0], None, [16], [0, 256])
                hist_g = cv2.calcHist([face_roi], [1], None, [16], [0, 256])
                hist_r = cv2.calcHist([face_roi], [2], None, [16], [0, 256])
                
                # 归一化并展平特征
                hist_b = cv2.normalize(hist_b, hist_b).flatten()
                hist_g = cv2.normalize(hist_g, hist_g).flatten()
                hist_r = cv2.normalize(hist_r, hist_r).flatten()
                
                # 合并颜色特征
                color_features = np.concatenate([hist_b, hist_g, hist_r])
                
                # 提取HOG特征（简化版）
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (64, 128))
                hog = cv2.HOGDescriptor()
                hog_features = hog.compute(resized_face).flatten()
                
                # 合并特征并调整到固定长度
                combined_features = np.concatenate([color_features, hog_features[:436]])  # 确保总长度为500
                observation = combined_features[:500]  # 截断或补零到500维
            else:
                # 如果没有检测到人脸，使用随机噪声作为观察
                observation = np.random.normal(0, 0.2, 500)
            
            # 保存当前帧用于显示
            self.current_frame = frame
            
            # 感知与训练流程
            processed_features = self.conscious.perceive(observation)
            self.conscious.exchange_data()
            
            # 更新共享记忆库
            if random.random() < 0.1:  # 10%概率更新
                best_net = max(self.conscious.networks, 
                            key=lambda n: np.mean(list(n.train_loss)[-10:]) if len(n.train_loss) > 0 else 0)
                if len(best_net.thought_stream) > 0:
                    self.conscious.shared_memory.append({
                        "data": best_net.thought_stream[-1],
                        "emotion": best_net.emotion,
                        "importance": np.mean(best_net.thought_stream[-1])
                    })
                    # 巩固重要记忆
                    self.conscious.consolidate_important_memory(
                        best_net.thought_stream[-1],
                        best_net.emotion,
                        np.mean(best_net.thought_stream[-1])
                    )
            
            # 各网络训练过程
            for i, net in enumerate(self.conscious.networks):
                processed, importance = processed_features[i]
                if self.prev_processed[i] is not None:
                    net.memory.append((self.prev_processed[i], processed, importance))
                self.prev_processed[i] = processed
                
                if len(net.memory) >= net.batch_size and net.alertness > 0.4:
                    # 确保有非零权重
                    weights = [t[2] for t in net.memory]
                    total_weight = sum(weights)
                    
                    # 处理权重全部为零的情况
                    if total_weight <= 0:
                        # 使用均匀权重替代
                        samples = random.sample(net.memory, k=net.batch_size)
                    else:
                        # 使用权重采样
                        samples = random.choices(
                            net.memory,
                            weights=weights,
                            k=net.batch_size
        )
                    
                    X = np.array([s[0] for s in samples])
                    y_next = np.array([s[1] for s in samples])
                    y_stats = np.array([[np.mean(n), np.std(n)] for n in y_next])
                    y = np.hstack([y_next, y_stats])
                    
                    # 增强特征：加入其他网络的连接节点
                    other_nodes = [n.connection_node for j, n in 
                                  enumerate(self.conscious.networks) if j != i]
                    X_aug = np.hstack([X, np.tile(other_nodes, (X.shape[0], 1))])
                    
                    # 首次训练或后续训练
                    if net.first_fit:
                        net.knowledge.fit(X_aug, y)
                        net.first_fit = False
                    else:
                        loss = net.knowledge.partial_fit(X_aug, y).loss_
                        net.train_loss.append(loss)
                        
                        # 自适应学习率调整
                        if len(net.train_loss) > 20:
                            recent_loss = np.mean(list(net.train_loss)[-10:])
                            avg_loss = np.mean(net.train_loss)
                            emotion_gain = 0.5 + 1.5 * net.emotion
                            if recent_loss < avg_loss * 0.9:
                                net.learning_rate = min(0.01, net.learning_rate * 1.05 * emotion_gain)
                            else:
                                net.learning_rate = max(0.0001, net.learning_rate * 0.95 / emotion_gain)
                            net.batch_size = int(np.clip(net.batch_size * (recent_loss/avg_loss)**0.5, 64, 512))
                            net.knowledge.learning_rate_init = net.learning_rate
                
                # 情绪自动调节系统
                self.auto_adjust_emotion(net)
                
                # 生成思维流
                if not net.first_fit and net.alertness > 0.3:
                    other_nodes = [n.connection_node for j, n in 
                                 enumerate(self.conscious.networks) if j != i]
                    input_data = np.hstack([processed, other_nodes])
                    thought = net.knowledge.predict(input_data.reshape(1, -1))
                    net.thought_stream.append(thought[0].tolist())
                    net.connection_node = thought[0][-1]  # 更新连接节点
                    
                    # 好奇心满足：新想法降低好奇心
                    if len(net.thought_stream) > 2:
                        novelty = np.mean(np.abs(np.array(net.thought_stream[-1]) - np.array(net.thought_stream[-2])))
                        net.curiosity = max(0.1, net.curiosity - 0.05 * novelty)
                    
                    if len(net.thought_stream) > 1000:
                        net.thought_stream.pop(0)
            
            time.sleep(0.05)
    
    def auto_adjust_emotion(self, net):
        """自动调整情绪值"""
        # 基于训练损失的调整
        if len(net.train_loss) > 10:
            recent_loss = np.mean(list(net.train_loss)[-5:])
            avg_loss = np.mean(net.train_loss)
            
            # 损失下降时提升情绪，损失上升时降低情绪
            loss_change = avg_loss - recent_loss
            loss_factor = np.tanh(loss_change * 5)  # 缩放变化幅度
            
            # 基于思维流波动的调整
            if len(net.thought_stream) > 50:
                thoughts = np.array(net.thought_stream[-50:])
                std_dev = np.std(thoughts)
                std_factor = np.tanh((std_dev - 0.1) * 2)  # 0.1作为基准波动
            else:
                std_factor = 0
            
            # 基于连接节点的调整
            conn_factor = np.tanh((net.connection_node - 0.5) * 2)
            
            # 基于警觉度的调整
            alert_factor = 2 * (net.alertness - 0.5)
            
            # 基于目标进度的调整
            goal_factor = 0
            if net.goal is not None:
                goal_factor = np.tanh((net.goal_progress - 0.5) * 2)
            
            # 综合调整因素
            adjustment = (0.4 * loss_factor + 0.2 * std_factor + 
                          0.1 * conn_factor + 0.2 * alert_factor + 0.1 * goal_factor)
            
            # 应用调整并确保在[0,1]范围内
            net.emotion = np.clip(net.emotion + 0.03 * adjustment, 0.1, 0.9)
        
        # 记录情绪变化
        net.emotion_history.append(net.emotion)
        if len(net.emotion_history) > 200:
            net.emotion_history.pop(0)
    
    def update_goals(self, net):
        """更新网络目标系统"""
        # 随机设定新目标
        if net.goal is None and random.random() < 0.02:
            goal_types = ["最小化损失", "最大化情感", "增强连接", "提高警觉", "识别人脸特征", "学习色彩模式"]
            net.goal = {
                "type": random.choice(goal_types),
                "created": time.time(),
                "target_value": random.uniform(0.5, 0.9)
            }
            net.goal_progress = 0.0
            net.curiosity = min(0.9, net.curiosity + 0.2)  # 新目标增加好奇心
        
        # 更新目标进度
        if net.goal is not None:
            if net.goal["type"] == "最小化损失":
                if len(net.train_loss) > 0:
                    current_value = np.mean(list(net.train_loss)[-5:])
                    net.goal_progress = 1 - min(1.0, current_value / net.goal["target_value"])
            elif net.goal["type"] == "最大化情感":
                net.goal_progress = net.emotion / net.goal["target_value"]
            elif net.goal["type"] == "增强连接":
                net.goal_progress = net.connection_node / net.goal["target_value"]
            elif net.goal["type"] == "提高警觉":
                net.goal_progress = net.alertness / net.goal["target_value"]
            elif net.goal["type"] == "识别人脸特征":
                # 基于是否检测到人脸来更新进度
                if self.current_frame is not None:
                    gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                    net.goal_progress = min(1.0, len(faces) * 0.3)
            elif net.goal["type"] == "学习色彩模式":
                # 基于颜色特征的多样性来更新进度
                if hasattr(self, 'current_features') and len(self.current_features) > 0:
                    color_variability = np.std(self.current_features[:48])  # 前48个是颜色特征
                    net.goal_progress = min(1.0, color_variability * 2)
            
            # 目标达成
            if net.goal_progress >= 0.95:
                net.emotion = min(0.9, net.emotion + 0.1)  # 目标达成提升情感
                net.goal = None
                net.curiosity = max(0.3, net.curiosity - 0.1)  # 目标达成降低好奇心
            
            # 目标放弃（长时间未达成）
            elif time.time() - net.goal["created"] > 30:  # 30秒未达成
                net.emotion = max(0.1, net.emotion - 0.05)  # 目标放弃降低情感
                net.goal = None
    
    def update_metacognition(self, net):
        """更新元认知能力"""
        # 元认知基于对自身表现的评估
        self_awareness = 0.0
        
        # 评估1：损失变化的认知
        if len(net.train_loss) > 10:
            recent_loss = np.mean(list(net.train_loss)[-5:])
            avg_loss = np.mean(net.train_loss)
            loss_awareness = 1.0 - min(1.0, abs(recent_loss - avg_loss) / (avg_loss + 1e-5))
            self_awareness += 0.4 * loss_awareness
        
        # 评估2：情感稳定性的认知
        if len(net.emotion_history) > 10:
            emotion_std = np.std(net.emotion_history[-10:])
            emotion_awareness = 1.0 - min(1.0, emotion_std / 0.2)
            self_awareness += 0.3 * emotion_awareness
        
        # 评估3：目标进展的认知
        if net.goal is not None:
            goal_awareness = net.goal_progress
            self_awareness += 0.3 * goal_awareness
        
        # 更新元认知（平滑过渡）
        net.metacognition = 0.9 * net.metacognition + 0.1 * self_awareness

    def start(self):
        threading.Thread(target=self.cognitive_cycle, daemon=True).start()
    
    def stop(self):
        """停止系统并释放资源"""
        self.running = False
        if hasattr(self, 'camera') and self.camera.isOpened():
            self.camera.release()

if __name__ == "__main__":
    mind = AutonomousMind()
    vis = Visualization(mind)
    mind.start()
    
    try:
        plt.show()
    finally:
        mind.stop()  # 确保程序结束时释放摄像头资源    