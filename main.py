"""
股票分析系统主入口
提供多种使用模式：单股票分析、批量分析、交互式流程等
"""
import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.stock_analysis_system import StockAnalysisSystem
from src.flows.investment_flow import SmartInvestmentFlow
from src.flows.batch_analysis_flow import BatchAnalysisFlow
from src.utils.llm_config import configure_llm_provider

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 首先配置默认LLM提供商（默认通义千问，可通过 LLM_PROVIDER 切换）
configure_llm_provider()

def check_environment():
    """检查环境配置"""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"错误：缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请在.env文件中设置这些变量")
        return False

    return True


def analyze_single_stock(company: str, ticker: str, use_cache: bool = True) -> Dict[str, Any]:
    """分析单只股票"""
    logger.info(f"开始分析单只股票: {company} ({ticker})")

    system = StockAnalysisSystem()
    result = system.analyze_stock(company, ticker, use_cache)

    if result['success']:
        print(f"\n✅ 分析成功: {company} ({ticker})")
        print(f"投资评级: {result['investment_rating']['rating']}")
        print(f"综合评分: {result['overall_score']:.1f}/100")
        print(f"报告路径: {result['report_path']}")
        print(f"数据路径: {result['data_path']}")
    else:
        print(f"\n❌ 分析失败: {result['error']}")

    return result


def analyze_multiple_stocks(stocks: List[Dict[str, str]], max_workers: int = 3):
    """批量分析多只股票"""
    logger.info(f"开始批量分析 {len(stocks)} 只股票")

    system = StockAnalysisSystem()
    results = system.analyze_multiple_stocks(stocks, max_workers)

    # 统计结果
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n📊 批量分析结果:")
    print(f"总股票数: {len(stocks)}")
    print(f"成功分析: {len(successful)}")
    print(f"失败分析: {len(failed)}")
    print(f"成功率: {len(successful)/len(stocks)*100:.1f}%")

    # 显示成功分析的股票
    if successful:
        print(f"\n✅ 成功分析的股票:")
        for result in successful:
            rating = result.get('investment_rating', {}).get('rating', '未评级')
            score = result.get('overall_score', 0)
            print(f"  - {result['company']} ({result['ticker']}): {rating} ({score:.1f}/100)")

    # 显示失败的股票
    if failed:
        print(f"\n❌ 失败的股票:")
        for result in failed:
            print(f"  - {result['company']} ({result['ticker']}): {result.get('error', '未知错误')}")

    # 生成摘要报告
    if successful:
        summary = system.generate_summary_report(results)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = f"reports/batch_summary_{timestamp}.md"

        os.makedirs('reports', exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"\n📄 摘要报告已保存: {summary_path}")

    return results


def run_interactive_flow():
    """运行交互式投资流程"""
    logger.info("启动交互式投资流程")

    flow = SmartInvestmentFlow()

    print("🚀 欢迎使用智能投资分析流程")
    print("系统将根据您的输入和分析结果智能调整分析深度")
    print("=" * 50)

    try:
        result = flow.kickoff()

        if result.get('success', False):
            print(f"\n🎉 分析流程成功完成!")
            print(f"公司: {result['company']} ({result['ticker']})")
            print(f"分析深度: {result['analysis_depth']}")
            print(f"综合评分: {result['overall_score']:.1f}/100")
            print(f"风险等级: {result['risk_level']}")
            print(f"投资建议: {result['recommendation']}")
            print(f"错误数量: {result['error_count']}")
            print(f"警告数量: {len(result['warnings'])}")
        else:
            print(f"\n❌ 分析流程失败: {result.get('error', '未知错误')}")

        # 显示状态摘要
        summary = flow.get_state_summary()
        print(f"\n📊 流程状态摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断流程")
    except Exception as e:
        logger.error(f"交互式流程异常: {str(e)}")
        print(f"\n❌ 流程异常: {str(e)}")


def run_batch_flow():
    """运行批量分析流程"""
    logger.info("启动批量分析流程")

    flow = BatchAnalysisFlow()

    print("📊 欢迎使用批量分析流程")
    print("系统将智能选择最佳的批量处理策略")
    print("=" * 50)

    try:
        result = flow.kickoff()

        if result.get('success', False):
            print(f"\n🎉 批量分析流程成功完成!")
            print(f"总股票数: {result['total_stocks']}")
            print(f"成功分析: {result['completed_count']}")
            print(f"失败分析: {result['failed_count']}")
            print(f"成功率: {result['success_rate']:.1f}%")
            print(f"摘要报告: {result['summary_path']}")

            # 显示详细状态
            status = flow.get_batch_status()
            print(f"\n📊 批量分析状态:")
            for key, value in status.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        else:
            print(f"\n❌ 批量分析流程失败: {result.get('error', '未知错误')}")

        # 显示详细错误信息
        detailed_results = flow.get_detailed_results()
        if detailed_results['errors']:
            print(f"\n❌ 错误详情:")
            for error in detailed_results['errors']:
                print(f"  - {error}")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断流程")
    except Exception as e:
        logger.error(f"批量流程异常: {str(e)}")
        print(f"\n❌ 流程异常: {str(e)}")


def show_system_info():
    """显示系统信息"""
    print("📋 股票分析系统信息")
    print("=" * 50)
    print("版本: 1.0.0")
    print("框架: CrewAI + Python")
    print("功能:")
    print("  • 智能股票分析")
    print("  • 多Agent协作")
    print("  • 流程控制")
    print("  • 批量处理")
    print("  • 报告生成")
    print("\n支持的命令:")
    print("  • single: 分析单只股票")
    print("  • batch: 批量分析")
    print("  • interactive: 交互式流程")
    print("  • batch-flow: 批量流程")
    print("  • info: 显示系统信息")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股票分析系统')
    parser.add_argument('command', nargs='?', choices=['single', 'batch', 'interactive', 'batch-flow', 'info'],
                       help='运行模式')
    parser.add_argument('--company', '-c', help='公司名称')
    parser.add_argument('--ticker', '-t', help='股票代码')
    parser.add_argument('--stocks-file', '-f', help='股票列表文件路径')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存')
    parser.add_argument('--max-workers', '-w', type=int, default=3, help='最大并发数')

    args = parser.parse_args()

    # 检查环境
    if not check_environment():
        return 1

    # 根据命令执行相应操作
    if args.command == 'single':
        if not args.company or not args.ticker:
            print("错误: 单股票分析需要指定 --company 和 --ticker 参数")
            return 1

        analyze_single_stock(args.company, args.ticker, not args.no_cache)

    elif args.command == 'batch':
        stocks = []

        if args.stocks_file:
            # 从文件读取股票列表
            try:
                with open(args.stocks_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(',')
                            if len(parts) == 2:
                                stocks.append({
                                    'company': parts[0].strip(),
                                    'ticker': parts[1].strip()
                                })
            except FileNotFoundError:
                print(f"错误: 文件不存在: {args.stocks_file}")
                return 1
        else:
            # 使用默认股票列表
            stocks = [
                {'company': '苹果公司', 'ticker': 'AAPL'},
                {'company': '微软', 'ticker': 'MSFT'},
                {'company': '谷歌', 'ticker': 'GOOGL'},
                {'company': '亚马逊', 'ticker': 'AMZN'},
                {'company': '特斯拉', 'ticker': 'TSLA'}
            ]

        if not stocks:
            print("错误: 没有找到要分析的股票")
            return 1

        analyze_multiple_stocks(stocks, args.max_workers)

    elif args.command == 'interactive':
        run_interactive_flow()

    elif args.command == 'batch-flow':
        run_batch_flow()

    elif args.command == 'info':
        show_system_info()

    else:
        # 没有指定命令，显示交互式菜单
        print("🎯 股票分析系统")
        print("=" * 50)
        print("请选择运行模式:")
        print("1. 单股票分析")
        print("2. 批量分析")
        print("3. 交互式投资流程")
        print("4. 批量分析流程")
        print("5. 系统信息")
        print("0. 退出")

        while True:
            try:
                choice = input("\n请输入选择 (0-5): ").strip()

                if choice == '0':
                    print("👋 再见!")
                    break
                elif choice == '1':
                    company = input("请输入公司名称: ").strip()
                    ticker = input("请输入股票代码: ").strip()
                    if company and ticker:
                        analyze_single_stock(company, ticker)
                elif choice == '2':
                    analyze_multiple_stocks([], args.max_workers)
                elif choice == '3':
                    run_interactive_flow()
                elif choice == '4':
                    run_batch_flow()
                elif choice == '5':
                    show_system_info()
                else:
                    print("❌ 无效选择，请重试")

            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                logger.error(f"交互式菜单异常: {str(e)}")
                print(f"❌ 发生错误: {str(e)}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
