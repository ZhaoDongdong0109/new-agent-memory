"""
LLM驱动的智能训练代理
基于大模型的自动化训练决策系统
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import requests


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # 本地模型
    DEEPSEEK = "deepseek"


@dataclass
class TrainingState:
    """当前训练状态"""
    stage: str
    epoch: int
    total_epochs: int
    loss: float
    accuracy: Optional[float]
    learning_rate: float
    batch_size: int
    gpu_memory_used: float
    gpu_memory_total: float
    training_time_hours: float
    recent_losses: List[float]
    errors: List[str]


@dataclass
class LLMAgentDecision:
    """智能体决策"""
    action: str  # "continue", "stop", "adjust_lr", "save", "analyze"
    confidence: float
    reasoning: str
    suggested_values: Dict[str, Any]


class LLMInference:
    """
    大模型推理接口
    支持多种LLM提供商
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4"
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.value.upper()}_API_KEY")
        self.base_url = base_url
        self.model = model
        
        if not self.api_key:
            print(f"⚠️ 警告: 未找到 {provider.value} API密钥")
            print("   设置环境变量或将密钥传入")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """生成回复"""
        if self.provider == LLMProvider.OPENAI:
            return self._openai_generate(prompt, max_tokens)
        elif self.provider == LLMProvider.ANTHROPIC:
            return self._anthropic_generate(prompt, max_tokens)
        elif self.provider == LLMProvider.DEEPSEEK:
            return self._deepseek_generate(prompt, max_tokens)
        elif self.provider == LLMProvider.LOCAL:
            return self._local_generate(prompt, max_tokens)
        else:
            raise ValueError(f"不支持的提供商: {self.provider}")
    
    def _openai_generate(self, prompt: str, max_tokens: int) -> str:
        """OpenAI API"""
        import openai
        openai.api_key = self.api_key
        
        if self.base_url:
            openai.api_base = self.base_url
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的AI训练专家，擅长分析和优化深度学习模型的训练过程。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _anthropic_generate(self, prompt: str, max_tokens: int) -> str:
        """Anthropic Claude API"""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def _deepseek_generate(self, prompt: str, max_tokens: int) -> str:
        """DeepSeek API"""
        if not self.base_url:
            self.base_url = "https://api.deepseek.com"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专业的AI训练专家。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _local_generate(self, prompt: str, max_tokens: int) -> str:
        """本地模型（如 vLLM, Ollama）"""
        if not self.base_url:
            self.base_url = "http://localhost:8000"
        
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=data
        )
        
        return response.json()["text"]


class IntelligentTrainingAgent:
    """
    基于LLM的智能训练代理
    
    功能：
    1. 分析训练状态
    2. 理解问题原因
    3. 提出智能建议
    4. 做出决策
    """
    
    def __init__(
        self,
        llm_inference: LLMInference,
        auto_mode: bool = True
    ):
        self.llm = llm_inference
        self.auto_mode = auto_mode
        
        # 训练历史
        self.analysis_history: List[Dict] = []
    
    def analyze_training_state(self, state: TrainingState) -> LLMAgentDecision:
        """
        分析当前训练状态并做出决策
        """
        # 构建分析提示
        prompt = self._build_analysis_prompt(state)
        
        print("\n" + "=" * 70)
        print("🤖 LLM 正在分析训练状态...")
        print("=" * 70)
        
        try:
            # 调用大模型
            response = self.llm.generate(prompt)
            
            # 解析决策
            decision = self._parse_decision(response)
            
            print(f"\n💭 LLM 分析结果:")
            print(f"   决策: {decision.action}")
            print(f"   置信度: {decision.confidence:.0%}")
            print(f"   原因: {decision.reasoning[:200]}...")
            
            # 保存分析历史
            self.analysis_history.append({
                "state": state.__dict__,
                "decision": decision.__dict__,
                "timestamp": time.time()
            })
            
            return decision
            
        except Exception as e:
            print(f"\n❌ LLM 调用失败: {e}")
            # 降级到规则决策
            return self._fallback_decision(state)
    
    def _build_analysis_prompt(self, state: TrainingState) -> str:
        """构建分析提示"""
        recent_losses_str = ", ".join([f"{l:.4f}" for l in state.recent_losses[-5:]])
        
        errors_str = "\n".join([f"- {e}" for e in state.errors[-3:]]) if state.errors else "无"
        
        prompt = f"""
# 训练状态分析请求

## 当前状态
- **训练阶段**: {state.stage}
- **当前轮次**: {state.epoch}/{state.total_epochs}
- **当前损失**: {state.loss:.6f}
- **准确率**: {state.accuracy:.2%}" if state.accuracy else "N/A"
- **学习率**: {state.learning_rate:.2e}
- **批次大小**: {state.batch_size}
- **GPU显存使用**: {state.gpu_memory_used:.1f}GB / {state.gpu_memory_total:.1f}GB
- **训练时间**: {state.training_time_hours:.2f}小时

## 最近损失变化
{recent_losses_str}

## 最近错误
{errors_str}

## 请分析以下问题

1. **损失趋势正常吗？** 损失在上升还是下降？速度如何？
2. **是否有问题？** 过拟合、欠拟合、梯度爆炸、内存不足？
3. **需要调整吗？** 学习率、批次大小、数据增强？
4. **应该继续还是停止？** 训练是否已经收敛？

## 输出格式

请按以下JSON格式输出你的决策：

```json
{{
    "action": "continue/adjust_lr/adjust_batch/save/stop",
    "confidence": 0.0-1.0,
    "reasoning": "你的分析原因（100-200字）",
    "suggested_values": {{
        "learning_rate": 数值或null,
        "batch_size": 数值或null,
        "other": "其他建议"
    }}
}}
```

请直接输出JSON，不要有其他内容。
"""
        return prompt
    
    def _parse_decision(self, response: str) -> LLMAgentDecision:
        """解析LLM的决策"""
        try:
            # 提取JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            data = json.loads(json_str.strip())
            
            return LLMAgentDecision(
                action=data.get("action", "continue"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "无"),
                suggested_values=data.get("suggested_values", {})
            )
            
        except Exception as e:
            print(f"解析失败: {e}，使用默认决策")
            return LLMAgentDecision(
                action="continue",
                confidence=0.0,
                reasoning="解析失败",
                suggested_values={}
            )
    
    def _fallback_decision(self, state: TrainingState) -> LLMAgentDecision:
        """降级到规则决策"""
        if len(state.recent_losses) >= 3:
            recent = state.recent_losses[-3:]
            if all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
                return LLMAgentDecision(
                    action="stop",
                    confidence=0.8,
                    reasoning="连续3轮损失未下降",
                    suggested_values={}
                )
        
        return LLMAgentDecision(
            action="continue",
            confidence=0.5,
            reasoning="规则决策：无明显问题，继续训练",
            suggested_values={}
        )
    
    def generate_training_report(self) -> str:
        """生成训练报告"""
        if not self.analysis_history:
            return "没有分析历史"
        
        prompt = f"""
# 训练报告生成

## 训练分析历史（共 {len(self.analysis_history)} 次分析）

"""
        
        for i, item in enumerate(self.analysis_history[-10:], 1):
            state = item["state"]
            decision = item["decision"]
            prompt += f"""
### 分析 {i}
- 阶段: {state['stage']}, 轮次: {state['epoch']}
- 损失: {state['loss']:.6f}
- 决策: {decision['action']} (置信度: {decision['confidence']:.0%})
- 原因: {decision['reasoning'][:100]}
"""
        
        prompt += """
请生成一份详细的训练报告，包括：
1. 整体训练表现评估
2. 关键转折点
3. 成功经验和失败教训
4. 对未来训练的建议

请用中文回答。
"""
        
        try:
            return self.llm.generate(prompt, max_tokens=2000)
        except Exception as e:
            return f"报告生成失败: {e}"


def create_llm_agent(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-4"
) -> IntelligentTrainingAgent:
    """创建LLM智能代理"""
    
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "anthropic": LLMProvider.ANTHROPIC,
        "deepseek": LLMProvider.DEEPSEEK,
        "local": LLMProvider.LOCAL
    }
    
    llm_provider = provider_map.get(provider.lower(), LLMProvider.OPENAI)
    
    llm = LLMInference(
        provider=llm_provider,
        api_key=api_key,
        model=model
    )
    
    return IntelligentTrainingAgent(llm)


if __name__ == "__main__":
    print("=" * 70)
    print("  🤖 LLM智能训练代理 - 演示")
    print("=" * 70)
    
    # 尝试创建代理
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    
    if api_key:
        print("\n✅ 找到API密钥，创建智能代理...")
        agent = create_llm_agent(api_key=api_key)
        
        # 模拟训练状态
        state = TrainingState(
            stage="encoder",
            epoch=10,
            total_epochs=50,
            loss=0.285,
            accuracy=0.92,
            learning_rate=1e-4,
            batch_size=32,
            gpu_memory_used=18.5,
            gpu_memory_total=24.0,
            training_time_hours=2.5,
            recent_losses=[0.320, 0.305, 0.295, 0.290, 0.285],
            errors=[]
        )
        
        print("\n📊 当前训练状态:")
        print(f"   损失: {state.loss:.6f}")
        print(f"   准确率: {state.accuracy:.2%}")
        print(f"   GPU显存: {state.gpu_memory_used:.1f}GB / {state.gpu_memory_total:.1f}GB")
        
        # 分析
        decision = agent.analyze_training_state(state)
        
        print(f"\n🎯 智能决策:")
        print(f"   行动: {decision.action}")
        print(f"   置信度: {decision.confidence:.0%}")
        print(f"   建议: {decision.suggested_values}")
        
    else:
        print("\n⚠️ 未找到API密钥")
        print("   请设置环境变量:")
        print("   export OPENAI_API_KEY=your_key")
        print("   或")
        print("   export DEEPSEEK_API_KEY=your_key")
        
        print("\n📝 当前实现状态:")
        print("   ✅ LLM推理接口已实现")
        print("   ✅ 智能分析逻辑已实现")
        print("   ✅ 支持多种LLM提供商")
        print("   ⏳ 需要API密钥才能真正运行")
    
    print("\n" + "=" * 70)
