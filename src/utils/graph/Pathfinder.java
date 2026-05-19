package utils.graph;

import utils.Vector2d;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Created by dperez on 13/01/16.
 */
public class Pathfinder
{
    public PathNode root;
    private NeighbourHelper provider;

    public HashSet<PathNode> nodes;
    private Map<Integer, PathNode> nodeCache;

    public Pathfinder(Vector2d rootPos, NeighbourHelper provider)
    {
        root = new PathNode(rootPos);
        this.provider = provider;
    }


    private ArrayList<PathNode> calculatePath(PathNode node)
    {
        ArrayDeque<PathNode> reversedPath = new ArrayDeque<>();
        while(node != null)
        {
            if(node.getParent() != null) //to avoid adding the start node.
            {
                reversedPath.addFirst(node);
            }
            node = node.getParent();
        }
        return new ArrayList<>(reversedPath);
    }

    //Dijkstraa to all possible destinations. Returns nodes of all destinations.
    public ArrayList<PathNode> findPaths()
    {
        return _dijkstra();
    }

    //A* to destination
    public ArrayList<PathNode> findPathTo(Vector2d goalPosition)
    {
        return _findPath(new PathNode(goalPosition));
    }


    private ArrayList<PathNode> _dijkstra()
    {
        resetSearchState();

        root.setVisited(true);
        root.setParent(null);
        root.setTotalCost(0.0);

        ArrayList<PathNode> destinationsFromStart = new ArrayList<>();
        PathNode node;

        PriorityQueue<PathNode> openList = new PriorityQueue<>();
        HashSet<PathNode> visited = new HashSet<>();
        PathNode rootNode = getCachedNode(root);
        visited.add(rootNode);
        openList.add(rootNode);

        while (openList.size() != 0)
        {
            node = openList.poll();

            if (node != rootNode)
            {
                destinationsFromStart.add(node);
            }

            ArrayList<PathNode> neighbours = provider.getNeighbours(node.getPosition(), node.getTotalCost());
            for (PathNode nb : neighbours) {
                double pathCost = nb.getTotalCost() + node.getTotalCost();
                PathNode neighbour = getCachedNode(nb);

                if (!visited.contains(neighbour)) {
                    neighbour.setVisited(true);
                    visited.add(neighbour);
                    neighbour.setTotalCost(pathCost);
                    openList.add(neighbour);
                } else if (pathCost < neighbour.getTotalCost()) {
                    neighbour.setTotalCost(pathCost);
                }
            }
        }

        return destinationsFromStart;
    }

    private ArrayList<PathNode> _findPath(PathNode goal)
    {
        // TODO this method repeats calculations that are already done in _dijsktra above, could be made a lot more
        // efficient to avoid re-calculating neighbours

        resetSearchState();
        PathNode node = null;
        PriorityQueue<PathNode> openList = new PriorityQueue<>();
        HashSet<PathNode> openSet = new HashSet<>();
        HashSet<PathNode> closedSet = new HashSet<>();

        root.setParent(null);
        root.setTotalCost(0.0);
        double dist = Vector2d.chebychevDistance(root.getPosition(), goal.getPosition());
        root.setEstimatedCost(dist);
        PathNode rootNode = getCachedNode(root);
        openList.add(rootNode);
        openSet.add(rootNode);

        while(openList.size() != 0)
        {
            node = openList.poll();
            if (!openSet.remove(node)) {
                continue;
            }
            closedSet.add(node);

            if(node.getX() == goal.getX() && node.getY() == goal.getY())
                return calculatePath(node);

            ArrayList<PathNode> neighbours = provider.getNeighbours(node.getPosition(), node.getTotalCost());

            for (PathNode nb : neighbours) {
                // This neighbour is a new object, it will not have any of the costs set up
                // use the cached nodes HashSet to find the correct object with the information available,
                // only missing estimated distance
                double pathCost = nb.getTotalCost() + node.getTotalCost();
                PathNode neighbour = getCachedNode(nb);
                boolean inOpen = openSet.contains(neighbour);
                boolean inClosed = closedSet.contains(neighbour);

                if (!inOpen && !inClosed) {
                    neighbour.setTotalCost(pathCost);
                    dist = Vector2d.chebychevDistance(neighbour.getPosition(), goal.getPosition());
                    neighbour.setEstimatedCost(dist);
                    neighbour.setParent(node);

                    openList.add(neighbour);
                    openSet.add(neighbour);

                } else if (pathCost < neighbour.getTotalCost()) {
                    neighbour.setTotalCost(pathCost);
                    neighbour.setParent(node);

                    closedSet.remove(neighbour);
                    if (inOpen) {
                        openSet.remove(neighbour);
                        openList.remove(neighbour);
                    }
                    openList.add(neighbour);
                    openSet.add(neighbour);
                }
            }

        }

        if(node == null || node.getX() != goal.getX() || node.getY() != goal.getY()) //not the goal
            return null;

        return calculatePath(node);

    }

    private void resetSearchState()
    {
        nodes = new HashSet<>();
        nodeCache = new HashMap<>();
    }

    private PathNode getCachedNode(PathNode candidate)
    {
        PathNode cached = nodeCache.get(candidate.getId());
        if (cached != null) {
            return cached;
        }

        nodeCache.put(candidate.getId(), candidate);
        nodes.add(candidate);
        return candidate;
    }




//    public void printPath(int pathId, ArrayList<Node> nodes)
//    {
//        if(nodes == null)
//        {
//            System.out.println("No Path");
//            return;
//        }
//
//        int[][] endsIds =  new int[2][2];
//
//        int org =  pathId / 10000;
//        int dest = pathId % 10000;
//
//        endsIds[0] = new int[]{org/100 , org%100};
//        endsIds[1] = new int[]{dest/100 , dest%100};
//
//        String ends = "(" + endsIds[0][0] + "," + endsIds[0][1] + ") -> ("
//                + endsIds[1][0] + "," + endsIds[1][1] + ")";
//
//
//        System.out.print("Path " + ends + "; ("+ nodes.size() + "): ");
//        for(Node n : nodes)
//        {
//            System.out.print(n.getX() + ":" + n.getY() + ", ");
//        }
//        System.out.println();
//    }
}
