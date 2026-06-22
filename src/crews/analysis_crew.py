"""
分析团队 - 使用CrewAI真正的多智能体协作
负责基本面分析、风险分析和行业分析，展示智能体间的深度协作与集体决策
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
from src.tools.fundamental_tools import FundamentalAnalysisTool
from src.tools.technical_tools import TechnicalAnalysisTool
from src.tools.financial_tools import FinancialCalculatorTool
from src.utils.http_utils import with_retry

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@CrewBase
class AnalysisCrew:
    """分析团队 - 展示深度协作分析和集体决策"""

    def __init__(self):
        """初始化分析团队"""
        # 显式传递 config_type 以兼容新版 CrewAI 的 _load_config 签名
        self.agents_config = self._load_config('config/agents.yaml', 'agents')
        self.tasks_config = self._load_config('config/tasks.yaml', 'tasks')

        # 如果配置文件加载失败，使用内置的默认配置
        if not self.agents_config:
            logger.warning("使用内置默认agents配置")
            self.agents_config = self._get_default_agents_config()
        if not self.tasks_config:
            logger.warning("使用内置默认tasks配置")
            self.tasks_config = self._get_default_tasks_config()

        logger.info(f"配置加载完成 - agents: {len(self.agents_config)}, tasks: {len(self.tasks_config)}")

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

    def _get_default_agents_config(self) -> Dict[str, Any]:
        """获取默认的agents配置"""
        return {
            "fundamental_analyst": {
                "role": "高级基本面分析师",
                "goal": "对{company}进行深度基本面分析，评估公司内在价值和长期投资潜力",
                "backstory": "你是一位经验丰富的基本面分析师，精通财务报表分析和价值评估。你能够深入分析公司的商业模式、竞争优势、财务健康状况和管理层质量，为投资决策提供可靠的基本面依据。"
            },
            "risk_assessment_expert": {
                "role": "风险评估专家",
                "goal": "全面评估{company}的投资风险，包括市场风险、财务风险和运营风险",
                "backstory": "你是专业的风险管理专家，擅长识别和量化各种投资风险。你能够全面分析公司面临的内外部风险因素，并提供专业的风险评估和建议。"
            },
            "industry_expert": {
                "role": "行业研究专家",
                "goal": "分析{company}所处行业的竞争格局、发展趋势和公司在行业中的地位",
                "backstory": "你是资深的行业研究专家，对行业发展趋势和竞争格局有深刻理解。你能够分析行业生命周期、市场容量、竞争强度和公司在行业中的竞争优势。"
            },
            "quantitative_analyst": {
                "role": "量化分析师",
                "goal": "通过量化方法验证{company}的分析结果，提供数据支持",
                "backstory": "你是专业的量化分析师，擅长使用统计和数学方法验证分析结果。你能够设计和应用量化模型来评估投资价值和风险。"
            },
            "analysis_coordinator": {
                "role": "分析协调员",
                "goal": "协调各分析师的工作，整合分析结果并形成最终投资建议",
                "backstory": "你是优秀的分析协调员，擅长整合多方观点和促进团队协作。你能够综合各个领域的分析结果，形成一致的投资判断和建议。"
            }
        }

    def _get_default_tasks_config(self) -> Dict[str, Any]:
        """获取默认的tasks配置"""
        return {
            "fundamental_analysis_task": {
                "description": "对{company}进行全面的基本面分析：1. 深度分析财务报表，包括利润表、资产负债表和现金流量表 2. 评估公司商业模式和竞争优势 3. 分析管理层质量和公司治理结构 4. 计算公司内在价值 5. 识别关键风险因素 6. 提供基本面投资判断",
                "expected_output": "详细的基本面分析报告，包含财务分析、价值评估、竞争优势分析和投资建议，输出为Markdown格式。"
            },
            "risk_assessment_task": {
                "description": "对{company}进行全面的风险评估：1. 市场风险分析（行业风险、竞争风险） 2. 财务风险分析（偿债能力、流动性风险） 3. 运营风险分析（管理风险、供应链风险） 4. 估值风险分析 5. 风险量化评估 6. 风险应对建议",
                "expected_output": "全面的风险评估报告，包含风险识别、量化评估和应对策略，输出为Markdown格式。"
            },
            "industry_analysis_task": {
                "description": "对{company}进行深度行业分析：1. 行业生命周期和发展阶段分析 2. 市场规模和增长趋势 3. 行业竞争格局和市场份额 4. 行业关键成功因素 5. 公司在行业中的竞争地位 6. 行业发展趋势和前景预测",
                "expected_output": "深度行业分析报告，包含行业分析、竞争地位评估和发展前景预测，输出为Markdown格式。"
            },
            "quantitative_validation_task": {
                "description": "对{company}进行量化验证分析：1. 建立量化估值模型 2. 历史数据回测验证 3. 敏感性分析 4. 统计显著性检验 5. 量化风险评估 6. 投资回报率预测",
                "expected_output": "量化验证报告，包含模型建立、数据验证和量化分析结果，输出为Markdown格式。"
            },
            "analysis_coordination_task": {
                "description": "协调和整合所有分析工作：1. 汇总各分析师的观点和结论 2. 识别和解决分析中的分歧 3. 综合各方分析形成投资建议 4. 评估建议的置信度 5. 提供最终投资判断和理由",
                "expected_output": "最终投资分析报告，包含综合分析结论、投资建议和详细理由，输出为Markdown格式。"
            }
        }

    @agent
    def fundamental_analyst(self) -> Agent:
        """基本面分析师 - 负责公司基本面深度分析"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'fundamental_analyst' in self.agents_config:
            return Agent(
                config=self.agents_config['fundamental_analyst'],
                verbose=True,
                tools=[FundamentalAnalysisTool(), FinancialCalculatorTool()],
                allow_delegation=True,  # 可以委托给风险或行业专家
                max_iter=8,
                memory=True,
                cache=True,
            )
        else:
            # 使用默认配置
            return Agent(
                role='高级基本面分析师',
                goal='对{company}进行深度基本面分析，评估公司内在价值和长期投资潜力',
                backstory="""你是一位经验丰富的基本面分析师，精通财务报表分析和价值评估。
                你能够深入分析公司的商业模式、竞争优势、财务健康状况和管理层质量，
                为投资决策提供可靠的基本面依据。""",
                verbose=True,
                tools=[FundamentalAnalysisTool(), FinancialCalculatorTool()],
                allow_delegation=True,  # 可以委托给风险或行业专家
                max_iter=8,
                memory=True,
                cache=True,
            )

    @agent
    def risk_assessment_specialist(self) -> Agent:
        """风险评估专家 - 负责识别和量化投资风险"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'risk_assessment_specialist' in self.agents_config:
            return Agent(
                config=self.agents_config['risk_assessment_specialist'],
                verbose=True,
                tools=[FinancialCalculatorTool(), TechnicalAnalysisTool()],
                allow_delegation=True,
                max_iter=8,
                memory=True,
                cache=True,
            )
        else:
            # 使用默认配置
            return Agent(
                role='风险评估专家',
                goal='识别和评估{company}的投资风险，提供风险量化分析',
                backstory="""你是一位专业的风险评估专家，擅长识别和分析投资风险。
                你能够从财务、经营、行业、宏观经济等多个维度评估风险，
                并提供量化的风险评估和风险管理建议。""",
                verbose=True,
                tools=[FinancialCalculatorTool(), TechnicalAnalysisTool()],
                allow_delegation=True,
                max_iter=8,
                memory=True,
                cache=True,
            )

    @agent
    def industry_expert(self) -> Agent:
        """行业专家 - 负责行业地位和竞争分析"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'industry_expert' in self.agents_config:
            return Agent(
                config=self.agents_config['industry_expert'],
                verbose=True,
                tools=[FundamentalAnalysisTool()],
                allow_delegation=True,
                max_iter=8,
                memory=True,
                cache=True,
            )
        else:
            # 使用默认配置
            return Agent(
                role='行业专家',
                goal='分析{company}在行业中的地位和竞争优势',
                backstory="""你是一位资深的行业专家，深入了解行业发展趋势和竞争格局。
                你能够分析公司在行业中的地位、竞争优势和市场份额，
                预测行业未来发展趋势和对公司的影响。""",
                verbose=True,
                tools=[FundamentalAnalysisTool()],
                allow_delegation=True,
                max_iter=8,
                memory=True,
                cache=True,
            )

    @agent
    def quantitative_analyst(self) -> Agent:
        """量化分析师 - 负责数据建模和统计验证"""
        return Agent(
            role='量化分析师',
            goal='使用数学和统计方法验证分析结论，构建预测模型',
            backstory="""你是一位资深的量化分析师，拥有数学和统计学背景。
            你擅长将复杂的市场数据转化为可量化的指标，构建预测模型，
            并使用统计方法验证其他分析师的结论。你相信数据驱动的决策，
            能够发现人类分析师可能忽略的模式和趋势。""",
            verbose=True,
            tools=[FinancialCalculatorTool(), TechnicalAnalysisTool()],
            allow_delegation=True,
            max_iter=8,
            memory=True,
            cache=True,
        )

    @agent
    def analysis_coordinator(self) -> Agent:
        """分析协调员 - 负责协调各分析师工作并整合结论"""
        return Agent(
            role='分析协调员',
            goal='协调各分析师的工作，整合分析结果，解决分析冲突，形成最终投资判断',
            backstory="""你是一位经验丰富的投资研究总监，擅长协调不同领域的专家。
            你能够理解基本面、风险、行业和量化分析的专业内容，
            识别不同分析师结论之间的差异和矛盾，促进专家间的讨论和辩论，
            最终形成一致的投资建议。你具有很强的综合判断能力和决策能力。""",
            verbose=True,
            allow_delegation=True,
            max_iter=10,
            memory=True,
            cache=True,
        )

    @task
    def fundamental_analysis_task(self) -> Task:
        """基本面分析任务 - 深度分析公司基本面"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'fundamental_analysis_task' in self.tasks_config:
            return Task(
                config=self.tasks_config['fundamental_analysis_task'],
                tools=[FundamentalAnalysisTool(), FinancialCalculatorTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='fundamental_analysis_report.md',
            )
        else:
            # 使用默认任务描述
            return Task(
                description="""
                对{company}进行全面的基本面分析：
                1. 深度分析财务报表，包括利润表、资产负债表和现金流量表
                2. 评估公司商业模式和竞争优势
                3. 分析管理层质量和公司治理结构
                4. 计算公司内在价值
                5. 识别关键风险因素
                6. 提供基本面投资判断
                """,
                expected_output="详细的基本面分析报告，包含财务分析、价值评估、竞争优势分析和投资建议，输出为Markdown格式。",
                tools=[FundamentalAnalysisTool(), FinancialCalculatorTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='fundamental_analysis_report.md',
            )

    @task
    def risk_assessment_task(self) -> Task:
        """风险评估任务 - 全面评估投资风险"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'risk_assessment_task' in self.tasks_config:
            return Task(
                config=self.tasks_config['risk_assessment_task'],
                tools=[FinancialCalculatorTool(), TechnicalAnalysisTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='risk_assessment_report.md',
            )
        else:
            # 使用默认任务描述
            return Task(
                description="""
                对{company}进行全面的风险评估：
                1. 市场风险分析（行业风险、竞争风险）
                2. 财务风险分析（偿债能力、流动性风险）
                3. 运营风险分析（管理风险、供应链风险）
                4. 估值风险分析
                5. 风险量化评估
                6. 风险应对建议
                """,
                expected_output="全面的风险评估报告，包含风险识别、量化评估和应对策略，输出为Markdown格式。",
                tools=[FinancialCalculatorTool(), TechnicalAnalysisTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='risk_assessment_report.md',
            )

    @task
    def industry_analysis_task(self) -> Task:
        """行业分析任务 - 分析行业地位和前景"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'industry_analysis_task' in self.tasks_config:
            return Task(
                config=self.tasks_config['industry_analysis_task'],
                tools=[FundamentalAnalysisTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='industry_analysis_report.md',
            )
        else:
            # 使用默认任务描述
            return Task(
                description="""
                对{company}进行深度行业分析：
                1. 行业生命周期和发展阶段分析
                2. 市场规模和增长趋势
                3. 行业竞争格局和市场份额
                4. 行业关键成功因素
                5. 公司在行业中的竞争地位
                6. 行业发展趋势和前景预测
                """,
                expected_output="深度行业分析报告，包含行业分析、竞争地位评估和发展前景预测，输出为Markdown格式。",
                tools=[FundamentalAnalysisTool()],
                context=[],  # 将在执行时动态设置
                human_input=False,
                output_file='industry_analysis_report.md',
            )

    @task
    def quantitative_validation_task(self) -> Task:
        """量化验证任务 - 验证分析结论的统计显著性"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'quantitative_validation_task' in self.tasks_config:
            return Task(
                config=self.tasks_config['quantitative_validation_task'],
                tools=[FinancialCalculatorTool(), TechnicalAnalysisTool()],
                context=[],  # 将在执行时动态设置所有前置任务
                human_input=False,
                output_file='quantitative_validation_report.md',
            )
        else:
            # 使用默认任务描述
            return Task(
                description="""
                使用量化方法验证基本面、风险和行业分析的结论：

                1. 构建统计模型验证预测的可靠性
                2. 计算各项指标的历史相关性
                3. 识别数据中的模式和异常值
                4. 评估分析师判断的统计置信度
                5. 提供量化化的投资建议

                公司: {company}
                股票代码: {ticker}
                分析数据: {analysis_data}
                """,
                expected_output="""
                量化验证报告，包含：
                - 统计显著性测试结果
                - 预测模型的准确性和置信区间
                - 数据模式识别结果
                - 量化投资建议和风险指标
                - 对其他分析师结论的统计评价
                """,
                tools=[FinancialCalculatorTool(), TechnicalAnalysisTool()],
                context=[],  # 将在执行时动态设置所有前置任务
                human_input=False,
                output_file='quantitative_validation_report.md',
            )

    @task
    def analysis_coordination_task(self) -> Task:
        """分析协调任务 - 整合所有分析结果并形成最终判断"""
        # 检查配置是否存在，如果不存在则使用默认配置
        if 'analysis_coordination_task' in self.tasks_config:
            return Task(
                config=self.tasks_config['analysis_coordination_task'],
                tools=[],  # 协调员主要使用分析和综合能力
                context=[],  # 将在执行时动态设置所有前置任务
                human_input=False,
                output_file='final_investment_analysis.md',
            )
        else:
            # 使用默认任务描述
            return Task(
                description="""
                协调和整合所有分析师的工作，形成最终的投资判断：

                1. 汇总基本面、风险、行业和量化分析的结果
                2. 识别各分析师结论之间的分歧和矛盾
                3. 促进分析师间的讨论和辩论
                4. 评估各项证据的权重和可靠性
                5. 解决分歧，形成一致的投资建议
                6. 提供投资决策的详细理由

                公司: {company}
                股票代码: {ticker}
                所有分析结果: {all_analysis_results}
                """,
                expected_output="""
                最终投资分析报告，包含：
                - 综合分析结论
                - 各分析师观点的整合
                - 分歧解决过程
                - 最终投资建议（买入/卖出/持有）
                - 建议的详细理由和依据
                - 关键风险因素和应对策略
                - 投资时间框架建议
                """,
                tools=[],  # 协调员主要使用分析和综合能力
                context=[],  # 将在执行时动态设置所有前置任务
                human_input=False,
                output_file='final_investment_analysis.md',
            )

    @property
    def agents(self) -> List[Agent]:
        """获取所有智能体"""
        return [
            self.fundamental_analyst(),
            self.risk_assessment_specialist(),
            self.industry_expert(),
            self.quantitative_analyst(),
            self.analysis_coordinator()
        ]

    @property
    def tasks(self) -> List[Task]:
        """获取所有任务"""
        return [
            self.fundamental_analysis_task(),
            self.risk_assessment_task(),
            self.industry_analysis_task(),
            self.quantitative_validation_task(),
            self.analysis_coordination_task()
        ]

    def create_crew(self) -> Crew:
        """创建Crew实例 - 配置深度分析协作"""
        return Crew(
            agents=self.agents,  # 所有分析师智能体
            tasks=self.tasks,    # 所有分析任务
            process=Process.hierarchical,  # 层次化协作，智能体自主讨论
            manager_llm='gpt-4o-mini',  # 管理者LLM，用于协调层次化流程
            verbose=True,
            memory=True,  # 启用团队记忆，保留分析过程
            cache=True,   # 启用缓存
            planning=True,  # 启用规划功能
            planning_llm='gpt-4o-mini',
            share_crew=True,  # 允许智能体间共享信息
        )

    @with_retry(max_retries=3, backoff_factor=1.0)
    def execute_collaborative_analysis(self, company: str, ticker: str,
                                    collection_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行多智能体协作分析，带自动重试机制"""
        try:
            logger.info(f"启动多智能体协作分析: {company} ({ticker})")

            # 准备分析数据
            analysis_inputs = self._prepare_analysis_inputs(company, ticker, collection_data)

            # 设置任务间的协作关系
            self._setup_analysis_collaboration()

            # 执行协作分析
            logger.info("开始执行多智能体协作分析...")
            crew_instance = self.create_crew()
            result = crew_instance.kickoff(inputs=analysis_inputs)

            # 收集各智能体的分析输出
            agent_outputs = self._collect_analysis_outputs()

            # 计算协作分析评分
            collaboration_scores = self._calculate_collaboration_scores(agent_outputs)

            # 生成最终投资建议
            final_recommendation = self._generate_final_recommendation(agent_outputs, collaboration_scores)

            logger.info(f"多智能体协作分析完成: {company}")

            return {
                'success': True,
                'company': company,
                'ticker': ticker,
                'result': result,
                'agent_outputs': agent_outputs,
                'collaboration_scores': collaboration_scores,
                'final_recommendation': final_recommendation,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'collaboration_metrics': self._analyze_collaboration_quality(agent_outputs)
            }

        except Exception as e:
            error_msg = f"多智能体协作分析失败: {str(e)}"
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

    def _prepare_analysis_inputs(self, company: str, ticker: str,
                               collection_data: Any = None) -> Dict[str, Any]:
        """准备分析输入数据"""
        inputs = {
            'company': company,
            'ticker': ticker,
            'analysis_data': '将在分析过程中填充具体数据',  # 为quantitative_validation_task提供占位符
            'all_analysis_results': '等待各分析师完成分析后将在此汇总结果'  # 为analysis_coordination_task提供占位符
        }

        # 如果有数据收集结果，提取关键信息
        if collection_data:
            # 处理 CrewOutput 对象或字典
            if hasattr(collection_data, 'result'):
                # CrewOutput 对象
                data_dict = collection_data.result if hasattr(collection_data.result, '__dict__') else collection_data.result
                if isinstance(data_dict, str):
                    # 如果 result 是字符串，直接使用
                    inputs['analysis_data'] = {'raw_data': data_dict}
                elif isinstance(data_dict, dict):
                    # 如果 result 是字典，提取关键信息
                    inputs['analysis_data'] = data_dict
                    financial_data = data_dict.get('financial_data', '')
                    if isinstance(financial_data, str) and len(financial_data) > 100:
                        inputs['financial_report'] = financial_data
                    market_research = data_dict.get('market_research', '')
                    if market_research:
                        inputs['market_research'] = market_research
                    technical_analysis = data_dict.get('technical_analysis', '')
                    if technical_analysis:
                        inputs['technical_analysis'] = technical_analysis
                else:
                    inputs['analysis_data'] = {'raw_data': str(data_dict)}
            elif isinstance(collection_data, dict):
                # 字典类型
                inputs['analysis_data'] = collection_data
                financial_data = collection_data.get('financial_data', '')
                if isinstance(financial_data, str) and len(financial_data) > 100:
                    inputs['financial_report'] = financial_data
                market_research = collection_data.get('market_research', '')
                if market_research:
                    inputs['market_research'] = market_research
                technical_analysis = collection_data.get('technical_analysis', '')
                if technical_analysis:
                    inputs['technical_analysis'] = technical_analysis
            else:
                # 其他类型，转换为字符串
                inputs['analysis_data'] = {'raw_data': str(collection_data)}

        return inputs

    def _setup_analysis_collaboration(self):
        """设置分析任务间的协作关系"""
        # 获取任务实例
        fundamental_task = self.fundamental_analysis_task()
        risk_task = self.risk_assessment_task()
        industry_task = self.industry_analysis_task()
        quant_task = self.quantitative_validation_task()
        coordination_task = self.analysis_coordination_task()

        # 量化验证依赖于所有基础分析任务
        quant_task.context = [fundamental_task, risk_task, industry_task]

        # 最终协调任务依赖于所有分析任务
        coordination_task.context = [fundamental_task, risk_task, industry_task, quant_task]

    def _collect_analysis_outputs(self) -> Dict[str, Any]:
        """收集各分析师的输出"""
        outputs = {}

        report_files = [
            'fundamental_analysis_report.md',
            'risk_assessment_report.md',
            'industry_analysis_report.md',
            'quantitative_validation_report.md',
            'final_investment_analysis.md'
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

    def _calculate_collaboration_scores(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """基于协作分析结果计算评分"""
        scores = {
            'fundamental_score': 0.0,
            'risk_score': 0.0,
            'industry_score': 0.0,
            'quantitative_score': 0.0,
            'overall_score': 0.0
        }

        try:
            # 从最终分析报告中提取评分
            final_report = outputs.get('final_investment_analysis.md', '')
            if final_report and "未生成" not in final_report:
                # 尝试从报告中提取评分信息
                import re

                # 查找评分模式
                score_patterns = {
                    'fundamental_score': r'基本面[评分得分].*?(\d+(?:\.\d+)?)',
                    'risk_score': r'风险[评分得分].*?(\d+(?:\.\d+)?)',
                    'industry_score': r'行业[评分得分].*?(\d+(?:\.\d+)?)',
                    'overall_score': r'综合[评分总分].*?(\d+(?:\.\d+)?)'
                }

                for score_type, pattern in score_patterns.items():
                    match = re.search(pattern, final_report)
                    if match:
                        scores[score_type] = float(match.group(1))

            # 如果从报告中无法提取评分，使用基于输出质量的估算
            if scores['overall_score'] == 0:
                quality_score = self._estimate_analysis_quality(outputs)
                scores['overall_score'] = quality_score

            # 确保分数在合理范围内
            for key in scores:
                scores[key] = max(0, min(100, scores[key]))

        except Exception as e:
            logger.error(f"计算协作评分时出错: {str(e)}")

        return scores

    def _estimate_analysis_quality(self, outputs: Dict[str, Any]) -> float:
        """基于输出质量估算分析评分"""
        quality_score = 60.0  # 基础分

        try:
            # 检查报告完整性
            complete_reports = 0
            total_reports = len(outputs)

            for file_name, content in outputs.items():
                if content and "未生成" not in content and "失败" not in content:
                    complete_reports += 1

                    # 检查内容质量
                    if len(content) > 500:  # 内容足够详细
                        quality_score += 5
                    if "分析" in content or "评估" in content:  # 包含分析内容
                        quality_score += 3
                    if "建议" in content or "结论" in content:  # 包含建议结论
                        quality_score += 2

            completeness_ratio = complete_reports / total_reports
            quality_score *= completeness_ratio

            return min(100, max(0, quality_score))

        except Exception as e:
            logger.error(f"估算分析质量时出错: {str(e)}")
            return 60.0

    def _generate_final_recommendation(self, outputs: Dict[str, Any],
                                     scores: Dict[str, float]) -> Dict[str, Any]:
        """生成最终投资建议"""
        recommendation = {
            'action': '持有',
            'confidence': 0.0,
            'reasoning': '',
            'time_horizon': '中期',
            'risk_level': '中等'
        }

        try:
            # 从最终报告中提取建议
            final_report = outputs.get('final_investment_analysis.md', '')
            if final_report and "未生成" not in final_report:
                # 提取投资建议
                if "买入" in final_report:
                    recommendation['action'] = "买入"
                    recommendation['confidence'] = 0.8
                elif "卖出" in final_report:
                    recommendation['action'] = "卖出"
                    recommendation['confidence'] = 0.8
                else:
                    recommendation['action'] = "持有"
                    recommendation['confidence'] = 0.6

                # 提取信心度和理由
                recommendation['reasoning'] = "基于多智能体协作分析的综合判断"

                # 根据评分调整信心度
                if scores['overall_score'] >= 80:
                    recommendation['confidence'] = min(1.0, recommendation['confidence'] + 0.2)
                elif scores['overall_score'] < 60:
                    recommendation['confidence'] = max(0.3, recommendation['confidence'] - 0.2)

        except Exception as e:
            logger.error(f"生成最终建议时出错: {str(e)}")

        return recommendation

    def _analyze_collaboration_quality(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """分析智能体协作质量"""
        metrics = {
            'total_agents': 5,
            'active_agents': 0,
            'collaboration_depth': 'low',
            'consensus_level': 0.0,
            'discussion_events': 0,
            'conflict_resolution': 0
        }

        try:
            # 分析各报告的协作迹象
            active_count = 0
            collaboration_indicators = 0

            for file_name, content in outputs.items():
                if content and "未生成" not in content and "失败" not in content:
                    active_count += 1

                    # 检查协作深度指标
                    if any(term in content for term in ["讨论", "辩论", "协商", "综合"]):
                        collaboration_indicators += 1
                    if "一致" in content or "共识" in content:
                        metrics['consensus_level'] += 0.25
                    if "分歧" in content or "差异" in content:
                        metrics['discussion_events'] += 1
                    if "解决" in content or "协调" in content:
                        metrics['conflict_resolution'] += 1

            metrics['active_agents'] = active_count

            # 评估协作深度
            if collaboration_indicators >= 3:
                metrics['collaboration_depth'] = 'high'
            elif collaboration_indicators >= 2:
                metrics['collaboration_depth'] = 'medium'

            # 计算共识水平
            metrics['consensus_level'] = min(1.0, metrics['consensus_level'])

        except Exception as e:
            logger.error(f"分析协作质量时出错: {str(e)}")

        return metrics

    def execute_analysis(self, company: str, ticker: str, collection_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行分析 - 兼容性方法，内部调用 execute_collaborative_analysis"""
        try:
            logger.info(f"执行分析: {company} ({ticker})")
            return self.execute_collaborative_analysis(company, ticker, collection_data)
        except Exception as e:
            logger.error(f"执行分析时出错: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "company": company,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }

    def calculate_analysis_score(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """计算分析评分 - 兼容性方法，供stock_analysis_system调用"""
        try:
            # 从analysis_data中提取协作结果
            if isinstance(analysis_data, dict):
                # 如果analysis_data包含协作结果，使用它
                if 'collaboration_scores' in analysis_data:
                    return analysis_data['collaboration_scores']
                elif 'scores' in analysis_data:
                    return analysis_data['scores']
                else:
                    # 如果没有评分数据，返回默认评分
                    return {
                        'fundamental_score': 70.0,
                        'risk_score': 70.0,
                        'industry_score': 70.0,
                        'quantitative_score': 70.0,
                        'overall_score': 70.0
                    }
            else:
                # 如果不是字典，返回默认评分
                return {
                    'fundamental_score': 70.0,
                    'risk_score': 70.0,
                    'industry_score': 70.0,
                    'quantitative_score': 70.0,
                    'overall_score': 70.0
                }
        except Exception as e:
            logger.error(f"计算分析评分时出错: {str(e)}")
            return {
                'fundamental_score': 60.0,
                'risk_score': 60.0,
                'industry_score': 60.0,
                'quantitative_score': 60.0,
                'overall_score': 60.0
            }

    def get_crew_info(self) -> Dict[str, Any]:
        """获取团队信息"""
        return {
            'name': '分析团队 (多智能体深度协作版)',
            'agents': [
                '基本面分析师',
                '风险评估专家',
                '行业专家',
                '量化分析师',
                '分析协调员'
            ],
            'description': '使用CrewAI实现深度协作分析，通过智能体讨论和辩论形成投资判断',
            'features': [
                '多维度分析协作',
                '智能体间讨论和辩论',
                '统计验证和量化支持',
                '集体决策和共识形成',
                '层次化分析和整合'
            ],
            'process': 'hierarchical (深度协作分析流程)'
        }
