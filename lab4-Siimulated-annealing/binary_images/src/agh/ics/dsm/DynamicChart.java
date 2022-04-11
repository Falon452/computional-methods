package agh.ics.dsm;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.Millisecond;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.xy.XYDataset;
import org.jfree.ui.ApplicationFrame;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

// https://localcoder.org/making-dynamic-line-chart-using-jfree-chart-in-java
public class DynamicChart extends ApplicationFrame implements ActionListener {
    private TimeSeries series;
    public double lastValue = 0;

    /**
     * Timer to refresh graph after every 1/4th of a second
     */
    private Timer timer = new Timer(250, this);


    public DynamicChart(final String title) {

        super(title);
        this.series = new TimeSeries("", Millisecond.class);

        final TimeSeriesCollection dataset = new TimeSeriesCollection(this.series);
        final JFreeChart chart = createChart(dataset);

        timer.setInitialDelay(1000);

        //Sets background color of chart
        chart.setBackgroundPaint(Color.LIGHT_GRAY);

        //Created JPanel to show graph on screen
        final JPanel content = new JPanel(new BorderLayout());

        //Created Chartpanel for chart area
        final ChartPanel chartPanel = new ChartPanel(chart);

        //Added chartpanel to main panel
        content.add(chartPanel);

        //Sets the size of whole window (JPanel)
        chartPanel.setPreferredSize(new java.awt.Dimension(800, 500));

        //Puts the whole content on a Frame
        setContentPane(content);

        timer.start();

    }

    private JFreeChart createChart(final XYDataset dataset) {
        final JFreeChart result = ChartFactory.createTimeSeriesChart(
                "Dynamic Line And TimeSeries Chart",
                "Time",
                "Value",
                dataset,
                true,
                true,
                false
        );

        final XYPlot plot = result.getXYPlot();

        plot.setBackgroundPaint(new Color(0xffffe0));
        plot.setDomainGridlinesVisible(true);
        plot.setDomainGridlinePaint(Color.lightGray);
        plot.setRangeGridlinesVisible(true);
        plot.setRangeGridlinePaint(Color.lightGray);

        ValueAxis xaxis = plot.getDomainAxis();
        xaxis.setAutoRange(true);

        //Domain axis would show data of 60 seconds for a time
        xaxis.setFixedAutoRange(120000.0);  // 60 seconds
        xaxis.setVerticalTickLabels(true);

        ValueAxis yaxis = plot.getRangeAxis();
        yaxis.setAutoRange(true);

        return result;
    }
    public void clear() {
        this.series.clear();
    }

    /**
     * Generates an random entry for a particular call made by time for every 1/4th of a second.
     *
     * @param e the action event.
     */
    public void actionPerformed(final ActionEvent e) {
        final Millisecond now = new Millisecond();
        this.series.add(new Millisecond(), this.lastValue);
    }

}