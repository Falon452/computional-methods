package agh.ics.dsm;

import agh.ics.dsm.Enums.Neighbourhood;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import javax.swing.JFrame;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import static java.lang.Integer.parseInt;

public class GUI extends JPanel implements ActionListener, ChangeListener {
	private static final long serialVersionUID = 1L;
	private Timer timer;
	private Board board;
	private JButton start;
	private JButton set;
	private JButton clear;
	private JComboBox<Neighbourhood> neighbourhoodType;
    private JLabel labelTemperature;
    private JLabel labelTemperatureMultiplier;
    private JLabel labelIterSpeed;
    private JLabel labelIterationsToCool;
    private JTextField temperature;
    private JTextField temperatureMultiplier;
    private JTextField iterationsToCool;
	private JSlider iterSpeed;
	private JFrame frame;
	private int iterNum = 0;
	private final int maxDelay = 500;
	private final int initDelay = 100;
	private boolean running = false;

	public GUI(JFrame jf) {
		frame = jf;
		timer = new Timer(initDelay, this);
		timer.stop();
	}

	public void initialize(Container container) {
		container.setLayout(new BorderLayout());
		container.setSize(new Dimension(1024, 768));

		JPanel buttonPanel = new JPanel();
		JPanel lowerButtonPanel = new JPanel();

		JPanel chartPanel = new JPanel();
        chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );


		start = new JButton("Start");
		start.setActionCommand("Start");
		start.addActionListener(this);

		clear = new JButton("clear");
		clear.setActionCommand("clear");
		clear.addActionListener(this);

        labelIterSpeed = new JLabel("Iteration speed");

		iterSpeed = new JSlider();
		iterSpeed.setMinimum(0);
		iterSpeed.setMaximum(maxDelay);
		iterSpeed.addChangeListener(this);
		iterSpeed.setValue(maxDelay - timer.getDelay());
		
		neighbourhoodType = new JComboBox<Neighbourhood>(Board.neighbourhoodTypes);
		neighbourhoodType.addActionListener(this);
		neighbourhoodType.setActionCommand("neighbourhoodType");

        labelTemperature = new JLabel("Temperature: ");

        temperature = new JTextField("1500", 6);
        temperature.addActionListener(this);
        temperature.setActionCommand("temperature");

        labelTemperatureMultiplier = new JLabel("Cooling:");

        temperatureMultiplier = new JTextField("0.99", 5);
        temperatureMultiplier.addActionListener(this);
        temperatureMultiplier.setActionCommand("temperatureMultiplier");

        labelIterationsToCool = new JLabel("iterations to cool:");

        iterationsToCool = new JTextField("5", 5);
        iterationsToCool.addActionListener(this);
        iterationsToCool.setActionCommand("iterationsToCool");

        set = new JButton("set");
        set.setActionCommand("set");
        set.addActionListener(this);


		buttonPanel.add(start);
		buttonPanel.add(clear);
        buttonPanel.add(labelIterSpeed);
        buttonPanel.add(iterSpeed);
		buttonPanel.add(neighbourhoodType);
        lowerButtonPanel.add(labelTemperature);
        lowerButtonPanel.add(temperature);
        lowerButtonPanel.add(labelTemperatureMultiplier);
        lowerButtonPanel.add(temperatureMultiplier);
        lowerButtonPanel.add(labelIterationsToCool);
        lowerButtonPanel.add(iterationsToCool);
        lowerButtonPanel.add(set);


		board = new Board(1024, 768 - buttonPanel.getHeight() - lowerButtonPanel.getHeight());
		container.add(board, BorderLayout.CENTER);
		container.add(buttonPanel, BorderLayout.SOUTH);
		container.add(lowerButtonPanel, BorderLayout.NORTH);
	}

    private DefaultCategoryDataset createDataset( ) {
        return new DefaultCategoryDataset( );
    }

	public void actionPerformed(ActionEvent e) {
		if (e.getSource().equals(timer)) {
			iterNum++;
			frame.setTitle("Binary Images, simulated annealing" + Integer.toString(iterNum) + " iteration)");
			board.iteration();
		} else {
			String command = e.getActionCommand();
			if (command.equals("Start")) {
				if (!running) {
					timer.start();
                    board.clearNeighbours();
                    board.initializeNeighbors(board.points);
					start.setText("Pause");
				} else {
					timer.stop();
					start.setText("Start");
				}
				running = !running;
                board.temperature = parseInt(temperature.getText());
				clear.setEnabled(true);

			} else if (command.equals("clear")) {
				iterNum = 0;
				timer.stop();
				start.setEnabled(true);
				board.clear();
				frame.setTitle("Cellular Automata Toolbox");
			}
			else if (command.equals("neighbourhoodType")) {
                board.selectedNeighbourhood = (Neighbourhood) neighbourhoodType.getSelectedItem();
            } else if (command.equals("temperature")) {
                board.temperature = Double.parseDouble(temperature.getText());
            } else if (command.equals("temperatureMultiplier")) {
                board.temperatureMultiplier = Double.parseDouble(temperatureMultiplier.getText());
            } else if (command.equals("iterationsToCool")) {
                board.iterationsToCool = Integer.parseInt(iterationsToCool.getText());
            } else if (command.equals("set")) {
                board.temperature = Double.parseDouble(temperature.getText());
                board.temperatureMultiplier = Double.parseDouble(temperatureMultiplier.getText());
                board.iterationsToCool = Integer.parseInt(iterationsToCool.getText());
            }
		}
	}

    public void temperatureTypes(KeyEvent e) {

    }

	public void stateChanged(ChangeEvent e) {
		timer.setDelay(maxDelay - iterSpeed.getValue());
	}
}
