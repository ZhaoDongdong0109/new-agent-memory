"""
训练数据生成器
生成用于训练记忆增强模型的数据
"""

import json
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import os


@dataclass
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str
    emotion: str = "neutral"
    importance: float = 0.5


@dataclass
class MemoryRecord:
    content: str
    memory_type: str  # "episodic", "semantic", "procedural", "emotional"
    importance: float
    emotion: str
    context: Dict[str, Any]


class TrainingDataGenerator:
    """
    训练数据生成器
    
    生成用于训练记忆模型的对话和记忆数据
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # 模板数据
        self.user_templates = [
            "我叫{name}，我喜欢{interest}",
            "我最近在{activity}",
            "我想{goal}",
            "今天{event}",
            "我遇到了{problem}",
            "我正在学习{subject}",
            "我对{topic}很感兴趣",
            "我需要{need}",
        ]
        
        self.assistant_templates = [
            "你好{name}！很高兴认识你。",
            "听起来很棒！{follow_up}",
            "我理解你的感受。",
            "让我来帮你。",
            "这是个好主意！",
            "继续加油！",
            "有什么我可以帮助你的吗？",
            "我记住了。",
        ]
        
        # 实体数据
        self.names = ["张三", "李四", "王五", "赵六", "小明", "小红", "小李", "小王"]
        self.interests = ["编程", "音乐", "运动", "阅读", "旅行", "美食", "电影", "游戏"]
        self.activities = ["学习Python", "健身", "看书", "写代码", "听音乐", "看电影", "逛街", "做饭"]
        self.goals = ["做一个项目", "学习新技能", "找到工作", "减肥", "存钱", "旅行", "交朋友"]
        self.subjects = ["Python", "机器学习", "英语", "数学", "历史", "科学", "艺术"]
        self.topics = ["科技", "文化", "历史", "自然", "社会", "经济", "健康"]
        self.needs = ["帮助", "建议", "信息", "支持", "指导", "资源"]
        self.events = ["天气很好", "工作很忙", "心情不错", "有点累", "很开心", "遇到了困难"]
        self.problems = ["一个技术问题", "一个选择困难", "一个挑战", "一个疑问"]
        
        self.emotions = ["joy", "neutral", "anticipation", "trust", "fear", "sadness"]
        self.memory_types = ["episodic", "semantic", "procedural", "emotional"]
    
    def _random_choice(self, items: List) -> str:
        return random.choice(items)
    
    def generate_conversation(self) -> Tuple[List[ConversationTurn], List[MemoryRecord]]:
        """生成一轮对话和对应的记忆"""
        turns = []
        memories = []
        
        # 随机选择对话模式
        pattern = random.choice(["introduction", "activity", "goal", "problem"])
        
        if pattern == "introduction":
            name = self._random_choice(self.names)
            interest = self._random_choice(self.interests)
            
            turns.append(ConversationTurn(
                role="user",
                content=f"我叫{name}，我喜欢{interest}",
                emotion="joy",
                importance=0.8
            ))
            
            turns.append(ConversationTurn(
                role="assistant",
                content=f"你好{name}！很高兴认识你。{interest}是很棒的兴趣！",
                emotion="joy",
                importance=0.6
            ))
            
            memories.append(MemoryRecord(
                content=f"用户叫{name}，喜欢{interest}",
                memory_type="episodic",
                importance=0.8,
                emotion="joy",
                context={"name": name, "interest": interest}
            ))
        
        elif pattern == "activity":
            activity = self._random_choice(self.activities)
            
            turns.append(ConversationTurn(
                role="user",
                content=f"我最近在{activity}",
                emotion="anticipation",
                importance=0.6
            ))
            
            turns.append(ConversationTurn(
                role="assistant",
                content=f"听起来很棒！{activity}是很有意义的活动。",
                emotion="trust",
                importance=0.5
            ))
            
            memories.append(MemoryRecord(
                content=f"用户最近在{activity}",
                memory_type="episodic",
                importance=0.6,
                emotion="anticipation",
                context={"activity": activity}
            ))
        
        elif pattern == "goal":
            goal = self._random_choice(self.goals)
            
            turns.append(ConversationTurn(
                role="user",
                content=f"我想{goal}",
                emotion="anticipation",
                importance=0.7
            ))
            
            turns.append(ConversationTurn(
                role="assistant",
                content=f"这是个好目标！我可以帮你{goal}。",
                emotion="trust",
                importance=0.6
            ))
            
            memories.append(MemoryRecord(
                content=f"用户想要{goal}",
                memory_type="episodic",
                importance=0.7,
                emotion="anticipation",
                context={"goal": goal}
            ))
        
        else:  # problem
            problem = self._random_choice(self.problems)
            
            turns.append(ConversationTurn(
                role="user",
                content=f"我遇到了{problem}",
                emotion="fear",
                importance=0.7
            ))
            
            turns.append(ConversationTurn(
                role="assistant",
                content=f"别担心，让我来帮你解决{problem}。",
                emotion="trust",
                importance=0.6
            ))
            
            memories.append(MemoryRecord(
                content=f"用户遇到了{problem}",
                memory_type="episodic",
                importance=0.7,
                emotion="fear",
                context={"problem": problem}
            ))
        
        return turns, memories
    
    def generate_multi_turn_conversation(self, num_turns: int = 3) -> Tuple[List[ConversationTurn], List[MemoryRecord]]:
        """生成多轮对话"""
        all_turns = []
        all_memories = []
        
        for _ in range(num_turns):
            turns, memories = self.generate_conversation()
            all_turns.extend(turns)
            all_memories.extend(memories)
        
        return all_turns, all_memories
    
    def generate_training_sample(self) -> Dict[str, Any]:
        """生成单个训练样本"""
        turns, memories = self.generate_multi_turn_conversation(random.randint(1, 5))
        
        return {
            "conversation": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "emotion": turn.emotion,
                    "importance": turn.importance
                }
                for turn in turns
            ],
            "memories": [
                {
                    "content": mem.content,
                    "type": mem.memory_type,
                    "importance": mem.importance,
                    "emotion": mem.emotion,
                    "context": mem.context
                }
                for mem in memories
            ]
        }
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        output_file: str = "training_data.json"
    ) -> List[Dict[str, Any]]:
        """生成完整数据集"""
        dataset = []
        
        print(f"生成 {num_samples} 个训练样本...")
        
        for i in range(num_samples):
            sample = self.generate_training_sample()
            dataset.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"  已生成 {i + 1}/{num_samples} 个样本")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"\n数据集已保存到: {output_file}")
        
        return dataset
    
    def generate_retrieval_pairs(
        self,
        num_pairs: int = 500,
        output_file: str = "retrieval_data.json"
    ) -> List[Dict[str, Any]]:
        """生成检索训练对（查询-记忆对）"""
        pairs = []
        
        print(f"生成 {num_pairs} 个检索训练对...")
        
        for i in range(num_pairs):
            # 生成一个对话
            turns, memories = self.generate_conversation()
            
            if memories:
                # 使用用户的问题作为查询
                query = turns[0].content
                
                # 正样本：正确的记忆
                positive_memory = memories[0].content
                
                # 负样本：随机选择其他记忆
                negative_memories = []
                for _ in range(3):  # 3个负样本
                    _, neg_memories = self.generate_conversation()
                    if neg_memories:
                        negative_memories.append(neg_memories[0].content)
                
                pairs.append({
                    "query": query,
                    "positive": positive_memory,
                    "negatives": negative_memories
                })
            
            if (i + 1) % 100 == 0:
                print(f"  已生成 {i + 1}/{num_pairs} 个检索对")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            print(f"\n检索数据已保存到: {output_file}")
        
        return pairs


def main():
    print("=" * 70)
    print("训练数据生成器")
    print("=" * 70)
    
    generator = TrainingDataGenerator()
    
    # 生成训练数据
    print("\n[1] 生成对话训练数据...")
    training_data = generator.generate_dataset(num_samples=1000, output_file="training_data.json")
    
    # 生成检索数据
    print("\n[2] 生成检索训练数据...")
    retrieval_data = generator.generate_retrieval_pairs(num_pairs=500, output_file="retrieval_data.json")
    
    # 统计
    print("\n" + "=" * 70)
    print("数据统计:")
    print(f"  对话样本数: {len(training_data)}")
    print(f"  检索对数: {len(retrieval_data)}")
    
    # 示例
    print("\n示例对话:")
    sample = training_data[0]
    for turn in sample["conversation"]:
        print(f"  [{turn['role']}]: {turn['content']}")
    
    print("\n对应记忆:")
    for mem in sample["memories"]:
        print(f"  - {mem['content']} (重要性: {mem['importance']})")
    
    print("\n" + "=" * 70)
    print("✅ 数据生成完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
