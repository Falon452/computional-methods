package agh.ics.dsm;

import java.util.ArrayList;

public class Point {

	public ArrayList<Point> neighbors;
	public int type;
    public Integer []neighbours;


	public Point() {
		type=0;
		neighbors= new ArrayList<Point>();
	}
	
	public void clear() {
        type = 0;
        neighbors.clear();
	}

    public void clearNeihbours() {
        neighbors.clear();
    }


	public void addNeighbor(Point nei) {
        neighbors.add(nei);
	}

    public int calculateEnergy() {
        int energy = 0;
        for (Point point : neighbors) {
            if (point.type == 1) {
                energy += 1;
            }
        }
        return energy;
    }
}