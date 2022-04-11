package agh.ics.dsm;

import java.awt.*;
import java.awt.event.*;


import agh.ics.dsm.Enums.Neighbourhood;
import org.jfree.ui.RefineryUtilities;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;


import javax.swing.*;
import javax.swing.event.MouseInputListener;

import static java.lang.Math.exp;
import static java.lang.System.exit;

public class Board extends JComponent implements MouseInputListener, ComponentListener {
	private static final long serialVersionUID = 1L;
	public Point[][] points;
	private int size = 10;
	public Neighbourhood selectedNeighbourhood = Neighbourhood.MooreRange1;
    public int totalEnergy = 1000;
    public double temperature = 1500;
    public double temperatureMultiplier = 0.99;
    public int iteration = 0;
    private final DynamicChart energyChart;
    private final DynamicChart tempChart;
    public int iterationsToCool = 5;

    private int length;
    private int height;


    static public Neighbourhood[] neighbourhoodTypes = {
            Neighbourhood.MooreRange1,
            Neighbourhood.MooreRange2,
            Neighbourhood.VonNeumannRange1,
            Neighbourhood.VonNeumannRange2,
            Neighbourhood.DiagonalRange1,
            Neighbourhood.DiagonalRange2,
            Neighbourhood.TopRowRange1,
            Neighbourhood.TopRowRange2,
            Neighbourhood.NE_E_NW_S,
            Neighbourhood.Smith,
            Neighbourhood.Cole
    };

	public Board(int length, int height) {
		addMouseListener(this);
		addComponentListener(this);
		addMouseMotionListener(this);
		setBackground(Color.WHITE);
		setOpaque(true);
        energyChart = new DynamicChart("Energy Chart");
        energyChart.pack();
        RefineryUtilities.positionFrameOnScreen(energyChart, 0, 1);
        energyChart.setVisible(true);

        tempChart = new DynamicChart("Temperature Chart");
        tempChart.pack();
        RefineryUtilities.positionFrameOnScreen(tempChart, 0.8, 1);
        tempChart.setVisible(true);
	}


	public void iteration() {

        //****** ARBITRARY SWAP
        int randomNumX = ThreadLocalRandom.current().nextInt(2, points.length - 2);
        int randomNumY = ThreadLocalRandom.current().nextInt(2, points[2].length - 2);

        while (points[randomNumX][randomNumY].type == 0) {
            randomNumX = ThreadLocalRandom.current().nextInt(2, points.length - 2);
            randomNumY = ThreadLocalRandom.current().nextInt(2, points[2].length - 2);
        }

        int shiftX = ThreadLocalRandom.current().nextInt(-1, 2);
        int shiftY = ThreadLocalRandom.current().nextInt(-1, 2);
        int xShifted = randomNumX + shiftX;
        int yShifted = randomNumY + shiftY;

        if (xShifted < 2 || xShifted >= points.length - 2) {
            xShifted = randomNumX;
        }
        if (yShifted < 2 || yShifted >= points[0].length - 2) {
            yShifted = randomNumY;
        }

        if (points[xShifted][yShifted].type == 0) {
            points[randomNumX][randomNumY].type = 0;
            points[xShifted][yShifted].type = 1;
        }

        int newTotalEnergy = 0;
        for (int x = 2; x < points.length - 2; ++x)
            for (int y = 2; y < points[x].length - 2; ++y) {
                if (points[x][y].type == 1) {
                    newTotalEnergy += points[x][y].calculateEnergy();
                }
            }

        if (newTotalEnergy > totalEnergy) {
            if (accept_worse(totalEnergy, newTotalEnergy, temperature)) {
                totalEnergy = newTotalEnergy;
            } else {
                points[randomNumX][randomNumY].type = 1;
                points[xShifted][yShifted].type = 0;
            }
        } else {
            totalEnergy = newTotalEnergy;
        }
        //****** END OF ARBITRARY SWAP

        iteration += 1;
        if (iteration % iterationsToCool == 0)
            temperature *= temperatureMultiplier;
        tempChart.lastValue = temperature;
        energyChart.lastValue = totalEnergy;
        this.repaint();
	}

	public void clear() {
        for (int x = 2; x < points.length - 2; ++x)
            for (int y = 2; y < points[x].length - 2; ++y)
                points[x][y].clear();


        energyChart.clear();
        tempChart.clear();
        this.repaint();
	}

	public void initialize() {
		points = new Point[this.length][this.height];

		for (int x = 0; x < points.length; ++x)
			for (int y = 0; y < points[x].length; ++y)
				points[x][y] = new Point();

        initializeNeighbors(this.points);
    }

    public void clearNeighbours() {
        for (int x = 2; x < points.length - 2; ++x)
            for (int y = 2; y < points[x].length - 2; ++y)
                points[x][y].clearNeihbours();
    }


    public void initializeNeighbors(Point[][] points){
        for (int x = 2; x < points.length-2; ++x) {
            for (int y = 2; y < points[x].length-2; ++y) {
                switch (this.selectedNeighbourhood) {
                    case MooreRange1 -> initializeMooreRange1(x, y);
                    case MooreRange2 -> initializeMooreRange2(x, y);
                    case VonNeumannRange1 -> initializeVonNeumannRange1(x, y);
                    case VonNeumannRange2 -> initializeVonNeumannRange2(x, y);
                    case DiagonalRange1 -> initializeDiagonalRange1(x, y);
                    case DiagonalRange2 -> initializeDiagonalRange2(x, y);
                    case TopRowRange1 -> initializeTopRowRange1(x, y);
                    case TopRowRange2 -> initializeTopRowRange2(x, y);
                    case NE_E_NW_S -> initialize_NE_E_NW_S(x, y);
                    case Smith -> initializeSmith(x, y);
                    case Cole -> initializeCole(x, y);
                }
            }
        }
    }

    private boolean accept_worse(int previous, int next, double temperature){
        if (next < previous) {
            exit(1);
            return false;
        }
        double probability = exp((previous - next) / temperature);
        Random rand = new Random();
        return rand.nextFloat() < probability;
    }

	protected void paintComponent(Graphics g) {
		if (isOpaque()) {
			g.setColor(getBackground());
			g.fillRect(0, 0, this.getWidth(), this.getHeight());
		}
		g.setColor(Color.GRAY);
		drawNetting(g, size);
	}

	private void drawNetting(Graphics g, int gridSpace) {
		Insets insets = getInsets();
		int firstX = insets.left;
		int firstY = insets.top;
		int lastX = this.getWidth() - insets.right;
		int lastY = this.getHeight() - insets.bottom;

		int x = firstX;
		while (x < lastX) {
			g.drawLine(x, firstY, x, lastY);
			x += gridSpace;
		}

		int y = firstY;
		while (y < lastY) {
			g.drawLine(firstX, y, lastX, y);
			y += gridSpace;
		}

		for (x = 2; x < points.length-2; ++x) {
			for (y = 2; y < points[x].length-2; ++y) {
				if (points[x][y].type==0){
					g.setColor(new Color(1.0f, 1.0f, 1.0f, 1f));
				}
				else if (points[x][y].type==1){
					g.setColor(new Color(0.0f, 0.0f, 0.0f, 1f));
				}

				g.fillRect((x * size) + 1, (y * size) + 1, (size - 1), (size - 1));
			}
		}
	}

	public void mouseClicked(MouseEvent e) {
		int x = e.getX() / size;
		int y = e.getY() / size;
		if ((x < points.length) && (x > 0) && (y < points[x].length) && (y > 0)) {
            points[x][y].type = 1;
			this.repaint();
		}
	}

	public void componentResized(ComponentEvent e) {
		int length = (this.getWidth() / size) + 1;
		int height = (this.getHeight() / size) + 1;
        this.length = length;
        this.height = height;
		initialize();
	}

	public void mouseDragged(MouseEvent e) {
		int x = e.getX() / size;
		int y = e.getY() / size;
		if ((x < points.length) && (x > 0) && (y < points[x].length) && (y > 0)) {
            points[x][y].type = 1;
			this.repaint();
		}


	}

    private void initializeMooreRange1(int x, int y) {
        points[x][y].addNeighbor(points[x - 1][y - 1]);
        points[x][y].addNeighbor(points[  x  ][y - 1]);
        points[x][y].addNeighbor(points[x + 1][y - 1]);
        points[x][y].addNeighbor(points[x - 1][  y  ]);
        points[x][y].addNeighbor(points[x + 1][  y  ]);
        points[x][y].addNeighbor(points[x - 1][y + 1]);
        points[x][y].addNeighbor(points[  x  ][y + 1]);
        points[x][y].addNeighbor(points[x + 1][y + 1]);
    }
    private void initializeMooreRange2(int x, int y) {
        initializeMooreRange1(x, y);
        for (int i = -1; i < 2; i++) {
            points[x][y].addNeighbor(points[x - 2][y + i]);
            points[x][y].addNeighbor(points[x + 2][y + i]);
            points[x][y].addNeighbor(points[x + i][y - 2]);
            points[x][y].addNeighbor(points[x + i][y + 2]);
        }
        points[x][y].addNeighbor(points[x + 2][y + 2]);
        points[x][y].addNeighbor(points[x - 2][y + 2]);
        points[x][y].addNeighbor(points[x + 2][y - 2]);
        points[x][y].addNeighbor(points[x + -2][y - 2]);
    }
    private void initializeVonNeumannRange1(int x, int y) {
        points[x][y].addNeighbor(points[  x  ][y - 1]);
        points[x][y].addNeighbor(points[  x  ][y + 1]);
        points[x][y].addNeighbor(points[x + 1][  y  ]);
        points[x][y].addNeighbor(points[x - 1][  y  ]);
    }
    private void initializeVonNeumannRange2(int x, int y) {
        initializeMooreRange1(x, y);
        points[x][y].addNeighbor(points[  x  ][y - 2]);
        points[x][y].addNeighbor(points[x + 1][y - 1]);
        points[x][y].addNeighbor(points[x + 2][  y  ]);
        points[x][y].addNeighbor(points[x + 1][y + 1]);
        points[x][y].addNeighbor(points[  x  ][y + 2]);
        points[x][y].addNeighbor(points[x - 1][y + 1]);
        points[x][y].addNeighbor(points[x - 2][  y  ]);
        points[x][y].addNeighbor(points[x - 1][y - 1]);
    }
    private void initializeDiagonalRange1(int x, int y) {
        points[x][y].addNeighbor(points[x - 1][y - 1]);
        points[x][y].addNeighbor(points[x + 1][y - 1]);
        points[x][y].addNeighbor(points[x + 1][y + 1]);
        points[x][y].addNeighbor(points[x - 1][y + 1]);
    }
    private void initializeDiagonalRange2(int x, int y) {
        initializeDiagonalRange1(x, y);
        points[x][y].addNeighbor(points[x - 2][y - 2]);
        points[x][y].addNeighbor(points[x - 2][y + 2]);
        points[x][y].addNeighbor(points[x + 2][y - 2]);
        points[x][y].addNeighbor(points[x + 2][y + 2]);
    }
    private void initializeTopRowRange1(int x, int y) {
        points[x][y].addNeighbor(points[x - 1][y - 1]);
        points[x][y].addNeighbor(points[  x  ][y - 1]);
        points[x][y].addNeighbor(points[x + 1][y - 1]);
    }
    private void initializeTopRowRange2(int x, int y) {
        initializeTopRowRange1(x, y);
        points[x][y].addNeighbor(points[x - 1][y - 2]);
        points[x][y].addNeighbor(points[  x  ][y - 2]);
        points[x][y].addNeighbor(points[x + 1][y - 2]);
    }

    private void initialize_NE_E_NW_S(int x, int y) {
        points[x][y].addNeighbor(points[x + 1][y + 1]);
        points[x][y].addNeighbor(points[x + 1][  y  ]);
        points[x][y].addNeighbor(points[x - 1][y + 1]);
        points[x][y].addNeighbor(points[  x  ][y - 1]);
    }
    private void initializeSmith(int x, int y) {
        points[x][y].addNeighbor(points[  x  ][y + 1]);
        points[x][y].addNeighbor(points[x - 1][y - 1]);
    }
    private void initializeCole(int x, int y) {
        points[x][y].addNeighbor(points[  x  ][y + 1]);
        points[x][y].addNeighbor(points[x + 1][y - 1]);
        points[x][y].addNeighbor(points[x - 1][y - 1]);
    }

	public void mouseExited(MouseEvent e) {
	}

	public void mouseEntered(MouseEvent e) {
	}

	public void componentShown(ComponentEvent e) {
	}

	public void componentMoved(ComponentEvent e) {
	}

	public void mouseReleased(MouseEvent e) {
	}

	public void mouseMoved(MouseEvent e) {
	}

	public void componentHidden(ComponentEvent e) {
	}

	public void mousePressed(MouseEvent e) {
	}

}

