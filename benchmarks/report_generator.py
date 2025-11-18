"""
Benchmark Report Generator

Generates comprehensive HTML reports and visualizations for benchmark results.
"""

import json
from typing import Dict, Any, List
from datetime import datetime


class BenchmarkReporter:
    """
    Generate comprehensive benchmark reports.
    """

    def __init__(self, results: Dict[str, Any]):
        """
        Initialize reporter with benchmark results.

        Args:
            results: Dictionary of benchmark results
        """
        self.results = results
        self.summary = results.get("summary", {})
        self.metadata = results.get("metadata", {})
        self.benchmark_results = results.get("results", [])
        self.comparisons = results.get("comparisons", [])

    def generate_html_report(self, output_file: str = "benchmark_report.html"):
        """Generate comprehensive HTML report."""
        html = self._generate_html()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

    def _generate_html(self) -> str:
        """Generate complete HTML report."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CandiApp Feature Engineering Benchmark Report</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_executive_summary()}
        {self._generate_performance_section()}
        {self._generate_quality_section()}
        {self._generate_ml_section()}
        {self._generate_comparison_section()}
        {self._generate_detailed_results()}
        {self._generate_footer()}
    </div>
</body>
</html>
"""

    def _get_css(self) -> str:
        """Get CSS styles for report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .section {
            padding: 40px;
            border-bottom: 1px solid #eee;
        }

        .section:last-child {
            border-bottom: none;
        }

        h2 {
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        h3 {
            color: #764ba2;
            font-size: 1.5em;
            margin-top: 30px;
            margin-bottom: 15px;
        }

        .executive-summary {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }

        .metric-card h4 {
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }

        .metric-unit {
            font-size: 0.9em;
            color: #666;
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .comparison-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }

        .comparison-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }

        .comparison-table tr:last-child td {
            border-bottom: none;
        }

        .comparison-table tr:hover {
            background: #f5f7fa;
        }

        .winner {
            color: #10b981;
            font-weight: bold;
        }

        .loser {
            color: #ef4444;
        }

        .tie {
            color: #f59e0b;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .badge-success {
            background: #10b981;
            color: white;
        }

        .badge-warning {
            background: #f59e0b;
            color: white;
        }

        .badge-danger {
            background: #ef4444;
            color: white;
        }

        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s;
        }

        footer {
            background: #f5f7fa;
            padding: 30px;
            text-align: center;
            color: #666;
        }

        .timestamp {
            font-size: 0.9em;
            color: #999;
            margin-top: 10px;
        }

        .highlight-box {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }

        .info-box {
            background: #dbeafe;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }

        .success-box {
            background: #d1fae5;
            border-left: 4px solid #10b981;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        </style>
"""

    def _generate_header(self) -> str:
        """Generate report header."""
        timestamp = self.metadata.get("timestamp", datetime.now().isoformat())
        samples = self.metadata.get("samples", "N/A")
        gpu_used = "Yes ✅" if self.metadata.get("gpu_used", False) else "No"

        return f"""
        <header>
            <h1>🎯 CandiApp Feature Engineering</h1>
            <p>Comprehensive Benchmark Report</p>
            <div class="timestamp">
                Generated: {timestamp}<br>
                Samples: {samples} | GPU: {gpu_used}
            </div>
        </header>
"""

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        comp_pos = self.summary.get("competitive_position", {})
        wins = comp_pos.get("wins", 0)
        ties = comp_pos.get("ties", 0)
        losses = comp_pos.get("losses", 0)
        win_rate = comp_pos.get("win_rate", 0)

        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="executive-summary">
                <h3>Overall Performance</h3>
                <p>
                    The CandiApp feature engineering system has been comprehensively benchmarked
                    against industry-leading ATS systems including RChilli, Textkernel, Sovren, and HireAbility.
                </p>

                <div class="metric-grid" style="margin-top: 20px;">
                    <div class="metric-card">
                        <h4>Competitive Wins</h4>
                        <div class="metric-value winner">{wins}</div>
                        <div class="metric-unit">vs competitors</div>
                    </div>
                    <div class="metric-card">
                        <h4>Ties</h4>
                        <div class="metric-value tie">{ties}</div>
                        <div class="metric-unit">comparable performance</div>
                    </div>
                    <div class="metric-card">
                        <h4>Win Rate</h4>
                        <div class="metric-value">{win_rate:.1f}%</div>
                        <div class="metric-unit">overall success</div>
                    </div>
                </div>

                <div class="progress-bar" style="margin-top: 30px;">
                    <div class="progress-fill" style="width: {win_rate}%">{win_rate:.0f}%</div>
                </div>
            </div>
        </div>
"""

    def _generate_performance_section(self) -> str:
        """Generate performance metrics section."""
        perf = self.summary.get("performance", {})

        metrics_html = ""
        for name, data in perf.items():
            value = data.get("value", 0)
            unit = data.get("unit", "")
            metrics_html += f"""
            <div class="metric-card">
                <h4>{name}</h4>
                <div class="metric-value">{value:.2f}</div>
                <div class="metric-unit">{unit}</div>
            </div>
"""

        return f"""
        <div class="section">
            <h2>⚡ Performance Benchmarks</h2>
            <div class="info-box">
                <strong>Note:</strong> Performance metrics measure speed, throughput, and resource efficiency
                of the feature engineering pipeline.
            </div>
            <div class="metric-grid">
                {metrics_html}
            </div>
        </div>
"""

    def _generate_quality_section(self) -> str:
        """Generate quality metrics section."""
        quality = self.summary.get("quality", {})

        metrics_html = ""
        for name, data in quality.items():
            value = data.get("value", 0)
            unit = data.get("unit", "")
            metrics_html += f"""
            <div class="metric-card">
                <h4>{name}</h4>
                <div class="metric-value">{value:.2f}</div>
                <div class="metric-unit">{unit}</div>
            </div>
"""

        return f"""
        <div class="section">
            <h2>🎯 Quality Metrics</h2>
            <div class="info-box">
                <strong>Note:</strong> Quality metrics assess feature coverage, completeness, and discriminability.
            </div>
            <div class="metric-grid">
                {metrics_html}
            </div>
        </div>
"""

    def _generate_ml_section(self) -> str:
        """Generate ML performance section."""
        ml = self.summary.get("ml", {})

        if not ml:
            return ""

        metrics_html = ""
        for name, data in ml.items():
            value = data.get("value", 0)
            unit = data.get("unit", "")
            metrics_html += f"""
            <div class="metric-card">
                <h4>{name}</h4>
                <div class="metric-value">{value:.2f}</div>
                <div class="metric-unit">{unit}</div>
            </div>
"""

        return f"""
        <div class="section">
            <h2>🤖 Machine Learning Performance</h2>
            <div class="success-box">
                <strong>ML Readiness:</strong> Features were tested with neural network classifiers
                to validate their effectiveness for downstream ML tasks.
            </div>
            <div class="metric-grid">
                {metrics_html}
            </div>
        </div>
"""

    def _generate_comparison_section(self) -> str:
        """Generate competitive comparison section."""
        rows_html = ""

        for comp in self.comparisons:
            system = comp["system"]
            metric = comp["metric"]
            our_score = comp["our_score"]
            competitor_score = comp["competitor_score"]
            winner = comp["winner"]
            improvement = comp["improvement_pct"]

            badge_class = "badge-success" if winner == "ours" else "badge-warning" if winner == "tie" else "badge-danger"
            winner_text = "✅ Us" if winner == "ours" else "🟡 Tie" if winner == "tie" else "❌ Them"

            rows_html += f"""
            <tr>
                <td><strong>{system}</strong></td>
                <td>{metric.replace('_', ' ').title()}</td>
                <td class="winner">{our_score:.2f}</td>
                <td>{competitor_score:.2f}</td>
                <td><span class="badge {badge_class}">{improvement:+.1f}%</span></td>
                <td>{winner_text}</td>
            </tr>
"""

        return f"""
        <div class="section">
            <h2>🏆 Competitive Comparison</h2>
            <div class="highlight-box">
                <strong>Industry Comparison:</strong> Direct comparison against leading ATS systems
                based on published benchmarks and industry standards.
            </div>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Competitor</th>
                        <th>Metric</th>
                        <th>Our Score</th>
                        <th>Their Score</th>
                        <th>Improvement</th>
                        <th>Winner</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
"""

    def _generate_detailed_results(self) -> str:
        """Generate detailed results table."""
        rows_html = ""

        for result in self.benchmark_results:
            name = result["name"]
            category = result["category"]
            metric = result["metric"]
            value = result["value"]
            unit = result["unit"]

            rows_html += f"""
            <tr>
                <td>{category.title()}</td>
                <td>{name}</td>
                <td>{metric.replace('_', ' ').title()}</td>
                <td><strong>{value:.3f}</strong> {unit}</td>
            </tr>
"""

        return f"""
        <div class="section">
            <h2>📋 Detailed Results</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Benchmark</th>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
"""

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return """
        <footer>
            <h3>CandiApp Feature Engineering System</h3>
            <p>
                This report was automatically generated by the CandiApp benchmarking suite.<br>
                For more information, visit the project repository.
            </p>
            <div class="timestamp">
                © 2024 CandiApp Team | AI-Powered Resume Analysis
            </div>
        </footer>
"""

    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*80)
        print("📊 BENCHMARK SUMMARY")
        print("="*80)

        # Performance
        if "performance" in self.summary:
            print("\n⚡ Performance:")
            for name, data in self.summary["performance"].items():
                print(f"  • {name}: {data['value']:.2f} {data['unit']}")

        # Quality
        if "quality" in self.summary:
            print("\n🎯 Quality:")
            for name, data in self.summary["quality"].items():
                print(f"  • {name}: {data['value']:.2f} {data['unit']}")

        # ML
        if "ml" in self.summary:
            print("\n🤖 Machine Learning:")
            for name, data in self.summary["ml"].items():
                print(f"  • {name}: {data['value']:.2f} {data['unit']}")

        # Competitive Position
        if "competitive_position" in self.summary:
            print("\n🏆 Competitive Position:")
            comp = self.summary["competitive_position"]
            print(f"  • Wins: {comp['wins']}")
            print(f"  • Ties: {comp['ties']}")
            print(f"  • Losses: {comp['losses']}")
            print(f"  • Win Rate: {comp['win_rate']:.1f}%")

        print("\n" + "="*80)
