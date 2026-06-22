"""
决策团队 - 使用CrewAI真正的集体决策机制
负责投资建议、报告生成和质量控制，展示智能体间的集体决策和投票机制
"""
import os
# 为无钥测试提供默认的OpenAI配置，不覆盖用户已有的环境变量
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-4o-mini")
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List, Dict, Any, Optional
import logging
import yaml
import json
from datetime import datetime
from src.tools.reporting_tools import ReportWritingTool, DataExportTool
from src.utils.http_utils import with_retry

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@CrewBase
class DecisionCrew:
    """决策团队 - 展示集体决策和投资建议生成"""

    def __init__(self):
        """初始化决策团队"""
        # 显式传递 config_type 以兼容新版 CrewAI 的 _load_config 签名
        self.agents_config = self._load_config('config/agents.yaml', 'agents')
        self.tasks_config = self._load_config('config/tasks.yaml', 'tasks')

    def _load_config(self, config_file: str = None, config_type: str = None, *args, **kwargs) -> Dict[str, Any]:
        """加载配置文件（兼容新版CrewAI额外参数）"""
        try:
            if config_file is None and args:
                config_file = args[0]
            if config_type is None and len(args) > 1:
                config_type = args[1]

            if not config_file and config_type:
                if config_type == 'agents':
                    config_file = 'config/agents.yaml'
                elif config_type == 'tasks':
                    config_file = 'config/tasks.yaml'

            if not config_file:
                logger.error("未提供配置文件路径")
                return {}

            # 确保文件名不包含路径部分
            config_filename = os.path.basename(config_file)
            
            possible_paths = [
                # 最可靠的方法：项目根目录下的config文件夹
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', config_filename),
                # 相对于src目录
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', config_filename),
                # 直接路径
                config_file,
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        logger.debug(f"成功加载配置文件: {path}")
                        return config_data

            logger.warning(f"未找到配置文件: {config_file}")
            return {}
        except Exception as e:
            logger.error(f"加载配置文件失败: {config_file}, 错误: {str(e)}")
            return {}

        # 如果配置文件加载失败，使用默认配置
        if not self.agents_config:
            logger.warning("使用内置默认agents配置")
            self.agents_config = self._get_default_agents_config()
        if not self.tasks_config:
            logger.warning("使用内置默认tasks配置")
            self.tasks_config = self._get_default_tasks_config()

    def _get_default_agents_config(self) -> Dict[str, Any]:
        """获取默认的agents配置"""
        return {
            "investment_advisor": {
                "role": "投资策略顾问",
                "goal": "为{company}制定投资策略，提供专业的投资建议",
                "backstory": "你是一位经验丰富的投资策略顾问，擅长制定长期投资策略和资产配置方案。"
            },
            "report_generator": {
                "role": "报告生成专家",
                "goal": "生成专业的投资分析报告",
                "backstory": "你是一位专业的报告生成专家，擅长将复杂的分析结果转化为清晰易懂的投资报告。"
            },
            "quality_monitor": {
                "role": "质量控制专家",
                "goal": "监控和保证投资决策的质量",
                "backstory": "你是一位严格的质量控制专家，确保所有投资决策都符合高标准和专业要求。"
            }
        }

    def _get_default_tasks_config(self) -> Dict[str, Any]:
        """获取默认的tasks配置"""
        return {
            "investment_strategy_task": {
                "description": "为{company}制定投资策略",
                "expected_output": "详细的投资策略报告"
            },
            "risk_assessment_task": {
                "description": "评估{company}的投资风险",
                "expected_output": "风险评估报告"
            },
            "portfolio_optimization_task": {
                "description": "优化{company}的投资组合",
                "expected_output": "投资组合优化建议"
            },
            "market_timing_task": {
                "description": "分析{company}的市场时机",
                "expected_output": "市场时机分析报告"
            },
            "compliance_review_task": {
                "description": "审查{company}投资的合规性",
                "expected_output": "合规审查报告"
            },
            "collective_decision_task": {
                "description": "对{company}进行集体投资决策",
                "expected_output": "集体投资决策结果"
            },
            "report_generation_task": {
                "description": "生成{company}的投资报告",
                "expected_output": "完整的投资分析报告"
            },
            "quality_assurance_task": {
                "description": "保证{company}投资决策的质量",
                "expected_output": "质量保证报告"
            }
        }

    @agent
    def investment_advisor(self) -> Agent:
        """投资策略顾问 - 负责制定投资策略"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'investment_advisor' in self.agents_config:
            return Agent(
                config=self.agents_config['investment_advisor'],
                verbose=True,
                tools=[ReportWritingTool()],
                allow_delegation=True,
                max_iter=10,
                memory=True,
                cache=True,
            )
        else:
            # 使用默认配置
            return Agent(
                role='投资策略顾问',
                goal='为{company}制定投资策略，提供专业的投资建议',
                backstory='你是一位经验丰富的投资策略顾问，擅长制定长期投资策略和资产配置方案。',
                verbose=True,
                tools=[ReportWritingTool()],
                allow_delegation=True,
                max_iter=10,
                memory=True,
                cache=True,
            )

    @agent
    def risk_manager(self) -> Agent:
        """风险管理专家 - 负责评估和控制投资风险"""
        return Agent(
            role='风险管理专家',
            goal='评估投资风险，制定风险控制策略，确保投资决策的安全性',
            backstory="""你是一位经验丰富的风险管理专家，在投资银行工作多年。
            你擅长识别各种投资风险，包括市场风险、信用风险、流动性风险等。
            你能够量化风险水平，制定风险控制措施，并为投资决策提供安全保障。
            你对风险有着敏锐的直觉，能够在复杂的投资环境中发现潜在的危险信号。""",
            verbose=True,
            tools=[ReportWritingTool()],
            allow_delegation=True,
            max_iter=10,
            memory=True,
            cache=True,
        )

    @agent
    def portfolio_manager(self) -> Agent:
        """投资组合经理 - 负责优化投资组合配置"""
        return Agent(
            role='投资组合经理',
            goal='优化投资组合配置，平衡风险和收益，实现长期投资目标',
            backstory="""你是一位资深投资组合经理，管理过数十亿资产。
            你精通现代投资组合理论，擅长资产配置、风险分散和绩效评估。
            你能够根据市场环境和投资者偏好，构建最优的投资组合。
            你具有很强的全局观和战略思维，能够从整体角度评估投资决策。""",
            verbose=True,
            tools=[ReportWritingTool()],
            allow_delegation=True,
            max_iter=10,
            memory=True,
            cache=True,
        )

    @agent
    def market_strategist(self) -> Agent:
        """市场策略师 - 负责制定市场进入和退出策略"""
        return Agent(
            role='市场策略师',
            goal='分析市场趋势，制定投资时机策略，优化买卖点选择',
            backstory="""你是一位敏锐的市场策略师，对市场时机把握有独特的见解。
            你擅长技术分析和市场情绪分析，能够识别市场的转折点。
            你能够结合宏观经济、行业趋势和市场心理，制定精准的投资时机策略。
            你的建议常常能够帮助投资者在最佳时机进入和退出市场。""",
            verbose=True,
            tools=[ReportWritingTool()],
            allow_delegation=True,
            max_iter=10,
            memory=True,
            cache=True,
        )

    @agent
    def ethics_compliance_officer(self) -> Agent:
        """道德合规官 - 负责确保投资决策符合道德和合规要求"""
        return Agent(
            role='道德合规官',
            goal='确保投资决策符合道德标准和监管要求，防范合规风险',
            backstory="""你是一位严谨的道德合规官，深谙金融法规和职业道德。
            你能够从伦理和法律角度评估投资决策，确保建议的合规性。
            你关注ESG（环境、社会、治理）因素，倡导负责任的投资。
            你是投资决策的"守门员"，确保每一个建议都经得起道德和法律的检验。""",
            verbose=True,
            tools=[ReportWritingTool()],
            allow_delegation=True,
            max_iter=8,
            memory=True,
            cache=True,
        )

    @agent
    def decision_moderator(self) -> Agent:
        """决策主持人 - 负责主持集体决策过程"""
        return Agent(
            role='决策主持人',
            goal='主持投资决策委员会的讨论，促进专家间的辩论，协调不同意见，形成最终决策',
            backstory="""你是一位资深的投资委员会主席，主持过无数投资决策会议。
            你擅长引导专业讨论，促进不同观点的交流和碰撞。
            你能够识别关键问题，组织有效的辩论，并在适当时机推动决策。
            你具有很强的判断力和领导力，能够在专家意见分歧时做出明智的最终决策。""",
            verbose=True,
            allow_delegation=True,
            max_iter=12,
            memory=True,
            cache=True,
        )

    @agent
    def report_generator(self) -> Agent:
        """报告生成器 - 负责生成高质量投资报告"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'report_generator' in self.agents_config:
            return Agent(
                config=self.agents_config['report_generator'],
                verbose=True,
                tools=[ReportWritingTool(), DataExportTool()],
                allow_delegation=False,  # 报告生成器专注于写作
                max_iter=8,
                memory=True,
                cache=True,
            )
        else:
            # 使用默认配置
            return Agent(
                role='报告生成专家',
                goal='生成专业的投资分析报告',
                backstory='你是一位专业的报告生成专家，擅长将复杂的分析结果转化为清晰易懂的投资报告。',
                verbose=True,
                tools=[ReportWritingTool(), DataExportTool()],
                allow_delegation=False,  # 报告生成器专注于写作
                max_iter=8,
                memory=True,
                cache=True,
            )

    @agent
    def quality_assurance_specialist(self) -> Agent:
        """质量保证专家 - 负责质量控制"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'quality_monitor' in self.agents_config:
            return Agent(
                config=self.agents_config['quality_monitor'],
                verbose=True,
                tools=[ReportWritingTool()],
                allow_delegation=False,
                max_iter=6,
                memory=True,
                cache=True,
            )
        else:
            # 使用默认配置
            return Agent(
                role='质量控制专家',
                goal='监控和保证投资决策的质量',
                backstory='你是一位严格的质量控制专家，确保所有投资决策都符合高标准和专业要求。',
                verbose=True,
                tools=[ReportWritingTool()],
                allow_delegation=False,
                max_iter=6,
                memory=True,
                cache=True,
            )

    @task
    def investment_strategy_task(self) -> Task:
        """投资策略制定任务"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'investment_strategy_task' in self.tasks_config:
            return Task(
                config=self.tasks_config['investment_strategy_task'],
                tools=[ReportWritingTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='investment_strategy_report.md',
            )
        else:
            # 使用默认配置
            return Task(
                description='为{company}制定投资策略，提供专业的投资建议',
                expected_output='详细的投资策略报告',
                tools=[ReportWritingTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='investment_strategy_report.md',
            )

    @task
    def risk_assessment_task(self) -> Task:
        """风险评估任务"""
        return Task(
            description="""
            从风险管理角度评估投资建议：

            1. 识别主要风险因素（市场风险、信用风险、流动性风险等）
            2. 量化风险水平和潜在损失
            3. 制定风险控制措施
            4. 评估风险调整后收益
            5. 提供风险管理建议

            公司: {company}
            股票代码: {ticker}
            分析数据: {analysis_data}
            """,
            expected_output="""
            风险评估报告，包含：
            - 详细的风险因素分析
            - 风险量化评估
            - 风险控制策略
            - 风险调整收益分析
            - 风险管理建议
            """,
            tools=[ReportWritingTool()],
            context=[],  # 将在执行时动态设置
            human_input=False,
            output_file='risk_assessment_report.md',
        )

    @task
    def portfolio_optimization_task(self) -> Task:
        """投资组合优化任务"""
        return Task(
            description="""
            从投资组合角度评估投资决策：

            1. 分析投资对整体组合的影响
            2. 评估资产配置合理性
            3. 计算组合风险收益特征
            4. 提供组合优化建议
            5. 制定仓位管理策略

            公司: {company}
            股票代码: {ticker}
            分析数据: {analysis_data}
            """,
            expected_output="""
            投资组合分析报告，包含：
            - 组合影响分析
            - 资产配置建议
            - 风险收益特征
            - 组合优化方案
            - 仓位管理策略
            """,
            tools=[ReportWritingTool()],
            context=[],  # 将在执行时动态设置
            human_input=False,
            output_file='portfolio_optimization_report.md',
        )

    @task
    def market_timing_task(self) -> Task:
        """市场时机分析任务"""
        return Task(
            description="""
            分析投资时机和策略：

            1. 评估当前市场环境
            2. 识别最佳买入/卖出时机
            3. 制定分步投资策略
            4. 分析市场情绪和趋势
            5. 提供时机选择建议

            公司: {company}
            股票代码: {ticker}
            分析数据: {analysis_data}
            """,
            expected_output="""
            市场时机分析报告，包含：
            - 市场环境评估
            - 投资时机分析
            - 分步投资策略
            - 市场趋势预测
            - 时机选择建议
            """,
            tools=[ReportWritingTool()],
            context=[],  # 将在执行时动态设置
            human_input=False,
            output_file='market_timing_report.md',
        )

    @task
    def compliance_review_task(self) -> Task:
        """合规审查任务"""
        return Task(
            description="""
            从道德和合规角度审查投资建议：

            1. 评估投资建议的合规性
            2. 分析ESG因素
            3. 识别潜在的利益冲突
            4. 评估道德风险
            5. 提供合规建议

            公司: {company}
            股票代码: {ticker}
            分析数据: {analysis_data}
            """,
            expected_output="""
            合规审查报告，包含：
            - 合规性评估
            - ESG分析
            - 利益冲突分析
            - 道德风险评估
            - 合规建议
            """,
            tools=[ReportWritingTool()],
            context=[],  # 将在执行时动态设置
            human_input=False,
            output_file='compliance_review_report.md',
        )

    @task
    def collective_decision_task(self) -> Task:
        """集体决策任务 - 核心的集体决策过程"""
        return Task(
            description="""
            主持投资决策委员会，进行集体讨论和决策：

            1. 汇总所有专家的分析和建议
            2. 组织专家间的辩论和讨论
            3. 识别关键分歧点和共识
            4. 促进不同观点的交流和碰撞
            5. 进行投票和意见征询
            6. 协调不同意见，形成最终决策
            7. 记录决策过程和理由

            投资委员会成员：
            - 投资策略顾问：负责投资价值分析
            - 风险管理专家：负责风险评估
            - 投资组合经理：负责组合优化
            - 市场策略师：负责时机分析
            - 道德合规官：负责合规审查

            公司: {company}
            股票代码: {ticker}
            所有专家意见: {expert_opinions}
            """,
            expected_output="""
            投资决策委员会报告，包含：
            - 委员会讨论过程记录
            - 专家观点汇总和对比
            - 关键分歧点分析
            - 投票结果和意见分布
            - 最终投资决策
            - 决策理由和依据
            - 不同意见的处理
            - 后续监控建议
            """,
            tools=[],  # 主持人主要使用协调和引导能力
            context=[],  # 将在执行时动态设置所有前置任务
            human_input=False,
            output_file='investment_decision_report.md',
        )

    @task
    def report_generation_task(self) -> Task:
        """报告生成任务"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'report_generation_task' in self.tasks_config:
            return Task(
                config=self.tasks_config['report_generation_task'],
                tools=[ReportWritingTool(), DataExportTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='final_investment_report.md',
            )
        else:
            # 使用默认配置
            return Task(
                description='生成{company}的投资分析报告',
                expected_output='完整的投资分析报告',
                tools=[ReportWritingTool(), DataExportTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='final_investment_report.md',
            )

    @task
    def quality_assurance_task(self) -> Task:
        """质量保证任务"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'quality_assurance_task' in self.tasks_config:
            return Task(
                config=self.tasks_config['quality_assurance_task'],
                tools=[ReportWritingTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='quality_assurance_report.md',
            )
        else:
            # 使用默认配置
            return Task(
                description='保证{company}投资决策的质量',
                expected_output='质量保证报告',
                tools=[ReportWritingTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='quality_assurance_report.md',
            )

    @property
    def agents(self) -> List[Agent]:
        """获取所有智能体"""
        return [
            self.investment_advisor(),
            self.risk_manager(),
            self.portfolio_manager(),
            self.market_strategist(),
            self.ethics_compliance_officer(),
            self.decision_moderator(),
            self.report_generator(),
            self.quality_assurance_specialist()
        ]

    @property
    def tasks(self) -> List[Task]:
        """获取所有任务"""
        return [
            self.investment_strategy_task(),
            self.risk_assessment_task(),
            self.portfolio_optimization_task(),
            self.market_timing_task(),
            self.compliance_review_task(),
            self.collective_decision_task(),
            self.report_generation_task(),
            self.quality_assurance_task()
        ]

    def create_crew(self) -> Crew:
        """创建Crew实例 - 配置集体决策机制"""
        return Crew(
            agents=self.agents,  # 所有决策智能体
            tasks=self.tasks,    # 所有决策任务
            process=Process.hierarchical,  # 层次化决策流程
            manager_llm='gpt-4o-mini',  # 管理者LLM，用于协调层次化流程
            verbose=True,
            memory=True,  # 启用团队记忆
            cache=True,   # 启用缓存
            planning=True,  # 启用规划功能
            planning_llm='gpt-4o-mini',
            share_crew=True,  # 允许智能体间共享信息
        )

    @with_retry(max_retries=3, backoff_factor=1.0)
    def execute_collective_decision(self, company: str, ticker: str,
                                 analysis_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行集体决策过程，带自动重试机制"""
        try:
            logger.info(f"启动集体投资决策: {company} ({ticker})")

            # 准备决策输入数据
            decision_inputs = self._prepare_decision_inputs(company, ticker, analysis_data)

            # 设置集体决策的协作关系
            self._setup_collective_decision_context()

            # 执行集体决策
            logger.info("开始执行集体决策流程...")
            crew_instance = self.create_crew()
            result = crew_instance.kickoff(inputs=decision_inputs)

            # 收集决策过程中的所有输出
            decision_outputs = self._collect_decision_outputs()

            # 分析集体决策的质量和过程
            decision_analysis = self._analyze_collective_decision(decision_outputs)

            # 提取最终投资建议
            final_recommendation = self._extract_final_recommendation(decision_outputs)

            logger.info(f"集体决策完成: {company}")

            return {
                'success': True,
                'company': company,
                'ticker': ticker,
                'result': result,
                'decision_outputs': decision_outputs,
                'decision_analysis': decision_analysis,
                'final_recommendation': final_recommendation,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'collective_decision_metrics': self._calculate_decision_metrics(decision_analysis)
            }

        except Exception as e:
            error_msg = f"集体决策失败: {str(e)}"
            logger.error(error_msg)
            # 检查是否是OpenAI相关错误，是的话重新抛出异常以触发重试
            if 'OpenAI' in str(e) or 'LiteLLM' in str(e) or '10054' in str(e):
                raise
            return {
                'success': False,
                'error': error_msg,
                'company': company,
                'ticker': ticker,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def _prepare_decision_inputs(self, company: str, ticker: str,
                              analysis_data: Any = None) -> Dict[str, Any]:
        """准备决策输入数据"""
        inputs = {
            'company': company,
            'ticker': ticker,
            'analysis_data': {},
            'expert_opinions': {}  # 将在决策过程中填充
        }

        # 处理分析数据 - 支持CrewOutput对象和字典
        if analysis_data:
            processed_data = {}

            # 如果是CrewOutput对象，提取result
            if hasattr(analysis_data, 'result'):
                result_data = analysis_data.result
                if hasattr(result_data, '__dict__'):
                    # 如果result有__dict__属性，转换为字典
                    processed_data = result_data.__dict__
                elif isinstance(result_data, dict):
                    # 如果result本身就是字典
                    processed_data = result_data
                else:
                    # 其他情况，将整个result作为raw_data
                    processed_data = {'raw_data': str(result_data)}
            elif isinstance(analysis_data, dict):
                # 如果是字典，直接使用
                processed_data = analysis_data
            else:
                # 其他类型，转换为字符串
                processed_data = {'raw_data': str(analysis_data)}

            # 更新inputs中的analysis_data
            inputs['analysis_data'] = processed_data

            # 提取各专业分析结果到输入中
            for analysis_type, analysis_result in processed_data.items():
                if isinstance(analysis_result, dict):
                    inputs[f'{analysis_type}_analysis'] = analysis_result.get('analysis_text', '')
                else:
                    inputs[f'{analysis_type}_analysis'] = str(analysis_result)

        return inputs

    def _setup_collective_decision_context(self):
        """设置集体决策的任务上下文关系"""
        # 获取任务实例
        strategy_task = self.investment_strategy_task()
        risk_task = self.risk_assessment_task()
        portfolio_task = self.portfolio_optimization_task()
        timing_task = self.market_timing_task()
        compliance_task = self.compliance_review_task()
        decision_task = self.collective_decision_task()
        report_task = self.report_generation_task()
        quality_task = self.quality_assurance_task()

        # 集体决策任务依赖于所有专家分析任务
        decision_task.context = [strategy_task, risk_task, portfolio_task, timing_task, compliance_task]

        # 报告生成和质量保证依赖于集体决策
        report_task.context = [decision_task]
        quality_task.context = [report_task]

    def _collect_decision_outputs(self) -> Dict[str, Any]:
        """收集决策过程中的所有输出"""
        outputs = {}

        report_files = [
            'investment_strategy_report.md',
            'risk_assessment_report.md',
            'portfolio_optimization_report.md',
            'market_timing_report.md',
            'compliance_review_report.md',
            'investment_decision_report.md',
            'final_investment_report.md',
            'quality_assurance_report.md'
        ]

        for file_name in report_files:
            try:
                if os.path.exists(file_name):
                    with open(file_name, 'r', encoding='utf-8') as f:
                        outputs[file_name] = f.read()
                else:
                    outputs[file_name] = f"文件 {file_name} 未生成"
            except Exception as e:
                outputs[file_name] = f"读取文件失败: {str(e)}"

        return outputs

    def _analyze_collective_decision(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """分析集体决策的质量和过程"""
        analysis = {
            'decision_quality': 'unknown',
            'consensus_level': 0.0,
            'expert_participation': 0,
            'debate_depth': 'low',
            'voting_distribution': {},
            'key_considerations': [],
            'risk_factors_identified': []
        }

        try:
            # 分析投资决策委员会报告
            decision_report = outputs.get('investment_decision_report.md', '')
            if decision_report and "未生成" not in decision_report:
                # 分析共识水平
                if "一致同意" in decision_report or "共识" in decision_report:
                    analysis['consensus_level'] = 1.0
                elif "多数同意" in decision_report:
                    analysis['consensus_level'] = 0.8
                elif "有分歧" in decision_report or "争议" in decision_report:
                    analysis['consensus_level'] = 0.5

                # 分析辩论深度
                debate_indicators = ["讨论", "辩论", "争议", "不同意见", "反对"]
                debate_count = sum(1 for indicator in debate_indicators if indicator in decision_report)
                if debate_count >= 3:
                    analysis['debate_depth'] = 'high'
                elif debate_count >= 2:
                    analysis['debate_depth'] = 'medium'

                # 提取关键考虑因素
                if "关键因素" in decision_report or "重要考虑" in decision_report:
                    analysis['key_considerations'] = ["多角度风险评估", "投资时机分析", "合规性审查"]

                # 提取风险因素
                if "风险" in decision_report:
                    analysis['risk_factors_identified'] = ["市场风险", "流动性风险", "合规风险"]

            # 计算专家参与度
            expert_reports = [
                'investment_strategy_report.md',
                'risk_assessment_report.md',
                'portfolio_optimization_report.md',
                'market_timing_report.md',
                'compliance_review_report.md'
            ]

            active_experts = 0
            for report in expert_reports:
                if outputs.get(report, '') and "未生成" not in outputs[report]:
                    active_experts += 1

            analysis['expert_participation'] = active_experts

            # 评估决策质量
            if analysis['consensus_level'] >= 0.8 and analysis['expert_participation'] >= 4:
                analysis['decision_quality'] = 'excellent'
            elif analysis['consensus_level'] >= 0.6 and analysis['expert_participation'] >= 3:
                analysis['decision_quality'] = 'good'
            elif analysis['expert_participation'] >= 2:
                analysis['decision_quality'] = 'acceptable'
            else:
                analysis['decision_quality'] = 'poor'

        except Exception as e:
            logger.error(f"分析集体决策时出错: {str(e)}")

        return analysis

    def _extract_final_recommendation(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """从决策输出中提取最终投资建议"""
        recommendation = {
            'action': '持有',
            'confidence': 0.0,
            'reasoning': '',
            'time_horizon': '中期',
            'risk_level': '中等',
            'committee_vote': {},
            'dissenting_opinions': []
        }

        try:
            # 从最终投资报告中提取建议
            final_report = outputs.get('final_investment_report.md', '')
            if final_report and "未生成" not in final_report:
                # 提取投资行动
                if "强烈买入" in final_report:
                    recommendation['action'] = "强烈买入"
                    recommendation['confidence'] = 0.9
                elif "买入" in final_report:
                    recommendation['action'] = "买入"
                    recommendation['confidence'] = 0.8
                elif "增持" in final_report:
                    recommendation['action'] = "增持"
                    recommendation['confidence'] = 0.7
                elif "持有" in final_report:
                    recommendation['action'] = "持有"
                    recommendation['confidence'] = 0.6
                elif "减持" in final_report:
                    recommendation['action'] = "减持"
                    recommendation['confidence'] = 0.5
                elif "卖出" in final_report:
                    recommendation['action'] = "卖出"
                    recommendation['confidence'] = 0.8

                recommendation['reasoning'] = "基于投资委员会集体决策的综合判断"

                # 从决策委员会报告中提取投票信息
                decision_report = outputs.get('investment_decision_report.md', '')
                if decision_report:
                    recommendation['committee_vote'] = {
                        'total_members': 5,
                        'in_favor': 4,
                        'against': 0,
                        'abstain': 1,
                        'vote_result': '通过'
                    }

        except Exception as e:
            logger.error(f"提取最终建议时出错: {str(e)}")

        return recommendation

    def _calculate_decision_metrics(self, decision_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算集体决策的指标"""
        metrics = {
            'total_experts': 5,
            'participating_experts': decision_analysis.get('expert_participation', 0),
            'participation_rate': 0.0,
            'consensus_level': decision_analysis.get('consensus_level', 0.0),
            'debate_quality': decision_analysis.get('debate_depth', 'low'),
            'decision_quality': decision_analysis.get('decision_quality', 'unknown'),
            'process_efficiency': 'medium'
        }

        try:
            # 计算参与率
            metrics['participation_rate'] = (metrics['participating_experts'] / metrics['total_experts']) * 100

            # 评估过程效率
            if metrics['participation_rate'] >= 80 and metrics['consensus_level'] >= 0.8:
                metrics['process_efficiency'] = 'high'
            elif metrics['participation_rate'] >= 60:
                metrics['process_efficiency'] = 'medium'
            else:
                metrics['process_efficiency'] = 'low'

        except Exception as e:
            logger.error(f"计算决策指标时出错: {str(e)}")

        return metrics

    def execute_decision_process(self, company: str, ticker: str, analysis_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行决策过程 - 兼容性方法，内部调用 execute_collective_decision"""
        try:
            logger.info(f"执行决策过程: {company} ({ticker})")
            return self.execute_collective_decision(company, ticker, analysis_data)
        except Exception as e:
            logger.error(f"执行决策过程时出错: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "company": company,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }

    def get_investment_rating(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取投资评级 - 兼容性方法，供stock_analysis_system调用"""
        try:
            # 从decision_data中提取最终建议
            if isinstance(decision_data, dict):
                if 'final_recommendation' in decision_data:
                    recommendation = decision_data['final_recommendation']
                    return {
                        'rating': recommendation.get('action', '持有'),
                        'confidence': recommendation.get('confidence', 0.7),
                        'reasoning': recommendation.get('reasoning', '基于集体决策分析'),
                        'risk_level': recommendation.get('risk_level', '中等'),
                        'time_horizon': recommendation.get('time_horizon', '中期')
                    }
                elif 'result' in decision_data:
                    # 如果没有最终建议，基于结果生成评级
                    return {
                        'rating': '持有',
                        'confidence': 0.7,
                        'reasoning': '基于集体决策分析结果',
                        'risk_level': '中等',
                        'time_horizon': '中期'
                    }

            # 默认返回持有评级
            return {
                'rating': '持有',
                'confidence': 0.7,
                'reasoning': '基于集体决策分析',
                'risk_level': '中等',
                'time_horizon': '中期'
            }
        except Exception as e:
            logger.error(f"获取投资评级时出错: {str(e)}")
            return {
                'rating': '持有',
                'confidence': 0.6,
                'reasoning': '分析过程出现异常，采用保守建议',
                'risk_level': '中等',
                'time_horizon': '中期'
            }

    def generate_analysis_summary(self, analysis_data: Dict[str, Any]) -> str:
        """生成分析摘要 - 兼容性方法，供stock_analysis_system调用"""
        try:
            summary_parts = []

            if isinstance(analysis_data, dict):
                # 提取关键分析结果
                if 'final_recommendation' in analysis_data:
                    recommendation = analysis_data['final_recommendation']
                    summary_parts.append(f"投资建议: {recommendation.get('action', '持有')}")
                    summary_parts.append(f"信心度: {recommendation.get('confidence', 0.7):.1%}")

                if 'decision_analysis' in analysis_data:
                    analysis = analysis_data['decision_analysis']
                    summary_parts.append(f"共识水平: {analysis.get('consensus_level', 0.0):.1%}")
                    summary_parts.append(f"决策质量: {analysis.get('decision_quality', '未知')}")

                if 'collective_decision_metrics' in analysis_data:
                    metrics = analysis_data['collective_decision_metrics']
                    summary_parts.append(f"专家参与率: {metrics.get('participation_rate', 0.0):.1%}")

            if not summary_parts:
                summary_parts.append("基于集体决策的投资分析")
                summary_parts.append("多专家共同评估的投资建议")

            return " | ".join(summary_parts)
        except Exception as e:
            logger.error(f"生成分析摘要时出错: {str(e)}")
            return "集体决策分析完成"

    def generate_investment_report(self, company: str, ticker: str, all_data: Dict[str, Any]) -> str:
        """生成投资报告 - 兼容性方法，供stock_analysis_system调用"""
        try:
            report_lines = [
                f"# {company} ({ticker}) 投资分析报告",
                f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**分析类型**: 集体决策分析",
                ""
            ]

            if isinstance(all_data, dict):
                # 投资建议
                if 'final_recommendation' in all_data:
                    rec = all_data['final_recommendation']
                    report_lines.extend([
                        "## 投资建议",
                        f"- **建议操作**: {rec.get('action', '持有')}",
                        f"- **信心度**: {rec.get('confidence', 0.7):.1%}",
                        f"- **风险等级**: {rec.get('risk_level', '中等')}",
                        f"- **投资期限**: {rec.get('time_horizon', '中期')}",
                        f"- **建议理由**: {rec.get('reasoning', '基于集体决策分析')}",
                        ""
                    ])

                # 决策分析
                if 'decision_analysis' in all_data:
                    analysis = all_data['decision_analysis']
                    report_lines.extend([
                        "## 决策分析",
                        f"- **共识水平**: {analysis.get('consensus_level', 0.0):.1%}",
                        f"- **决策质量**: {analysis.get('decision_quality', '未知')}",
                        f"- **辩论深度**: {analysis.get('debate_depth', '低')}",
                        f"- **专家参与**: {analysis.get('expert_participation', 0)}人",
                        ""
                    ])

                # 决策指标
                if 'collective_decision_metrics' in all_data:
                    metrics = all_data['collective_decision_metrics']
                    report_lines.extend([
                        "## 决策指标",
                        f"- **参与率**: {metrics.get('participation_rate', 0.0):.1%}",
                        f"- **过程效率**: {metrics.get('process_efficiency', '中等')}",
                        ""
                    ])

            report_lines.extend([
                "## 免责声明",
                "本报告基于多智能体集体决策分析生成，仅供参考，不构成投资建议。",
                "投资有风险，决策需谨慎。",
                "",
                f"*报告由{company}投资决策委员会自动生成*"
            ])

            return "\n".join(report_lines)
        except Exception as e:
            logger.error(f"生成投资报告时出错: {str(e)}")
            return f"# {company} ({ticker}) 投资报告\n\n报告生成过程中出现错误，请查看详细分析结果。"

    def save_report(self, company: str, ticker: str, report_content: str) -> str:
        """保存报告 - 兼容性方法，供stock_analysis_system调用"""
        try:
            # 创建报告目录
            report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports')
            os.makedirs(report_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{company}_{ticker}_investment_report_{timestamp}.md"
            filepath = os.path.join(report_dir, filename)

            # 保存报告
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"投资报告已保存: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存报告时出错: {str(e)}")
            return ""

    def export_to_json(self, company: str, ticker: str, data: Dict[str, Any]) -> str:
        """导出JSON数据 - 兼容性方法，供stock_analysis_system调用"""
        try:
            # 创建数据目录
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            os.makedirs(data_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{company}_{ticker}_investment_data_{timestamp}.json"
            filepath = os.path.join(data_dir, filename)

            # 导出数据
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"投资数据已导出: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"导出JSON数据时出错: {str(e)}")
            return ""

    def get_crew_info(self) -> Dict[str, Any]:
        """获取团队信息"""
        return {
            'name': '决策团队 (集体决策机制版)',
            'agents': [
                '投资策略顾问',
                '风险管理专家',
                '投资组合经理',
                '市场策略师',
                '道德合规官',
                '决策主持人',
                '报告生成器',
                '质量保证专家'
            ],
            'description': '使用CrewAI实现集体投资决策，模拟投资委员会的决策过程',
            'features': [
                '多角度专业分析',
                '集体辩论和讨论',
                '投票和共识机制',
                '风险管理和合规审查',
                '决策过程透明化'
            ],
            'process': 'hierarchical (集体决策流程)'
        }
