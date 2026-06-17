import argparse
import sys
import json
from autocorrelation_service import AutocorrelationService


def main():
    parser = argparse.ArgumentParser(
        description='时间序列自相关分析服务 - 计算 ACF 和 PACF'
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-f', '--file',
        type=str,
        help='输入数据文件路径（支持 .csv, .json, .txt）'
    )
    input_group.add_argument(
        '-d', '--data',
        type=str,
        help='直接输入数据，逗号分隔的数值列表'
    )
    input_group.add_argument(
        '--stdin',
        action='store_true',
        help='从标准输入读取数据'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='显著性水平 (默认: 0.05)'
    )
    parser.add_argument(
        '--nlags',
        type=int,
        default=None,
        help='最大滞后阶数 (默认: 数据长度的 1/4，最大 40)'
    )
    parser.add_argument(
        '--column',
        type=str,
        default='0',
        help='CSV 文件中的列名或列索引 (默认: 0)'
    )
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='CSV 文件没有表头'
    )
    parser.add_argument(
        '--no-stationarity',
        action='store_true',
        help='不执行平稳性检验'
    )
    parser.add_argument(
        '--no-ljung-box',
        action='store_true',
        help='不执行 Ljung-Box 白噪声检验'
    )
    parser.add_argument(
        '--lb-lags',
        type=str,
        default=None,
        help='Ljung-Box 检验的滞后阶数，可以是单个整数或逗号分隔的列表 (默认: 与 nlags 相同)'
    )
    parser.add_argument(
        '--lb-boxpierce',
        action='store_true',
        help='同时执行 Box-Pierce 检验'
    )
    parser.add_argument(
        '--json-output',
        action='store_true',
        help='以 JSON 格式输出结果'
    )
    parser.add_argument(
        '--json-indent',
        type=int,
        default=2,
        help='JSON 输出缩进 (默认: 2)'
    )

    args = parser.parse_args()

    service = AutocorrelationService(default_alpha=args.alpha)

    include_stationarity = not args.no_stationarity
    include_ljung_box = not args.no_ljung_box

    lb_lags = None
    if args.lb_lags:
        if ',' in args.lb_lags:
            lb_lags = [int(x.strip()) for x in args.lb_lags.split(',')]
        else:
            lb_lags = int(args.lb_lags)

    lb_boxpierce = args.lb_boxpierce

    try:
        if args.data:
            data = [float(x.strip()) for x in args.data.split(',')]
            result = service.analyze_from_list(
                data,
                alpha=args.alpha,
                nlags=args.nlags,
                include_stationarity=include_stationarity,
                include_ljung_box=include_ljung_box,
                ljung_box_lags=lb_lags,
                ljung_box_boxpierce=lb_boxpierce
            )
            result['success'] = True

        elif args.file:
            try:
                column = int(args.column)
            except ValueError:
                column = args.column

            result = service.analyze_from_file(
                args.file,
                column=column,
                has_header=not args.no_header,
                alpha=args.alpha,
                nlags=args.nlags,
                include_stationarity=include_stationarity,
                include_ljung_box=include_ljung_box,
                ljung_box_lags=lb_lags,
                ljung_box_boxpierce=lb_boxpierce
            )

        elif args.stdin:
            input_data = sys.stdin.read().strip()

            try:
                parsed = json.loads(input_data)
                if isinstance(parsed, dict) and 'data' in parsed:
                    result = service.analyze_from_json(
                        input_data,
                        alpha=args.alpha,
                        nlags=args.nlags,
                        include_stationarity=include_stationarity,
                        include_ljung_box=include_ljung_box,
                        ljung_box_lags=lb_lags,
                        ljung_box_boxpierce=lb_boxpierce
                    )
                elif isinstance(parsed, list):
                    result = service.analyze_from_list(
                        parsed,
                        alpha=args.alpha,
                        nlags=args.nlags,
                        include_stationarity=include_stationarity,
                        include_ljung_box=include_ljung_box,
                        ljung_box_lags=lb_lags,
                        ljung_box_boxpierce=lb_boxpierce
                    )
                    result['success'] = True
                else:
                    print("错误: 标准输入格式不正确", file=sys.stderr)
                    sys.exit(1)
            except json.JSONDecodeError:
                try:
                    data = [float(line.strip()) for line in input_data.splitlines()
                            if line.strip()]
                    result = service.analyze_from_list(
                        data,
                        alpha=args.alpha,
                        nlags=args.nlags,
                        include_stationarity=include_stationarity,
                        include_ljung_box=include_ljung_box,
                        ljung_box_lags=lb_lags,
                        ljung_box_boxpierce=lb_boxpierce
                    )
                    result['success'] = True
                except ValueError as e:
                    print(f"错误: 无法解析输入数据 - {e}", file=sys.stderr)
                    sys.exit(1)

        if not result.get('success', True):
            print(f"错误: {result.get('error', '未知错误')}", file=sys.stderr)
            sys.exit(1)

        if args.json_output:
            print(service.to_json(result, indent=args.json_indent))
        else:
            print(service.to_pretty_text(result))

    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
