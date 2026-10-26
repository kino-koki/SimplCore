(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // --- Chart: Tiny 配置损失曲线 ---
  var tinyLoss = [6.909, 6.389, 6.101, 5.878, 5.637, 5.444, 5.267, 5.102, 4.935, 4.765];
  var chart1 = echarts.init(document.getElementById('chart-tiny-loss'), null, { renderer: 'svg' });
  chart1.setOption({
    animation: false,
    tooltip: { trigger: 'axis', appendToBody: true },
    grid: { left: 50, right: 30, top: 30, bottom: 40 },
    xAxis: {
      type: 'category',
      data: tinyLoss.map(function(_, i) { return 'Step ' + (i + 1); }),
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: 'Loss',
      min: 4.5,
      max: 7.5,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [{
      name: 'Training Loss',
      type: 'line',
      data: tinyLoss,
      smooth: true,
      symbol: 'circle',
      symbolSize: 6,
      lineStyle: { color: accent, width: 2 },
      itemStyle: { color: accent },
      areaStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: accent + '40' },
            { offset: 1, color: accent + '05' }
          ]
        }
      },
      markPoint: {
        data: [
          { type: 'min', name: 'Min', symbolSize: 40 }
        ],
        itemStyle: { color: accent2 }
      }
    }]
  });
  window.addEventListener('resize', function() { chart1.resize(); });

  // --- Chart: 150M 配置损失曲线 ---
  var loss150m = [10.553, 8.417, 7.096, 5.924, 4.761];
  var chart2 = echarts.init(document.getElementById('chart-150m-loss'), null, { renderer: 'svg' });
  chart2.setOption({
    animation: false,
    tooltip: { trigger: 'axis', appendToBody: true },
    grid: { left: 50, right: 30, top: 30, bottom: 40 },
    xAxis: {
      type: 'category',
      data: loss150m.map(function(_, i) { return 'Step ' + (i + 1); }),
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: 'Loss',
      min: 3,
      max: 12,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [{
      name: 'Training Loss',
      type: 'line',
      data: loss150m,
      smooth: true,
      symbol: 'circle',
      symbolSize: 8,
      lineStyle: { color: accent2, width: 2 },
      itemStyle: { color: accent2 },
      areaStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: accent2 + '40' },
            { offset: 1, color: accent2 + '05' }
          ]
        }
      },
      markPoint: {
        data: [
          { type: 'min', name: 'Min', symbolSize: 40 }
        ],
        itemStyle: { color: accent }
      }
    }]
  });
  window.addEventListener('resize', function() { chart2.resize(); });
})();
