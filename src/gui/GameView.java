package gui;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.util.*;

import core.Types;
import core.actions.cityactions.CityAction;
import core.actions.unitactions.*;
import core.actions.unitactions.command.AttackCommand;
import core.actors.Actor;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Catapult;
import core.actors.units.SuperUnit;
import core.actors.units.Unit;
import core.game.Board;
import core.game.Game;
import core.game.GameState;
import core.actions.Action;
import utils.file.ImageIO;
import utils.Pair;
import utils.Vector2d;

import static core.Constants.*;
import static core.Types.TERRAIN.*;
import static core.Types.UNIT.*;
import static core.Types.ACTION.*;
import static utils.Vector2d.manhattanDistance;

@SuppressWarnings({"SuspiciousNameCombination", "unchecked"})
public class GameView extends JComponent {

    static int gridSize;
    private static final double TILE_WIDTH_SCALE = 1.55;
    private static final double TILE_HEIGHT_RATIO = 621.0 / 1019.0;
    private static final Map<Image, Rectangle> opaqueBoundsCache = new IdentityHashMap<>();
    private static final Map<Image, Point2D.Double> unitAnchorCache = new IdentityHashMap<>();
    private Game game;
    private Board board; //This only counts terrains. Needs to be enhanced with actors, resources, etc.
    private GameState gameState;
//    private Image backgroundImg;
    private Image fogImg, shineImg;
    private InfoView infoView;
    private Vector2d panTranslate;  // Used to translate all coordinates for objects drawn on screen

    private Color progressColor = new Color(53, 183, 255);
    private Color negativeColor = new Color(255, 63, 73);
    private Image starImg, starShadow, capitalImg, capitalShadow, cityWalls;
    private static final int[][] ROAD_DIRECTIONS = {
            {-1, -1}, {-1, 0}, {-1, 1},
            {0, -1},           {0, 1},
            {1, -1},  {1, 0},  {1, 1}
    };

    boolean[][] actionable;

    // Action animations
    private ArrayList<Pair<Pair<Image, Vector2d>,Pair<Image, Vector2d>>> sourceTargetAnimationInfo;
    private ArrayList<Double> animationSpeed;
    private ArrayList<Pair<Integer, Integer>> actionAnimationUnitsTribe;
    private UnitAction animatedAction;

    private Image[] explosionEffect, pierceEffect;
    private Image[][] slashEffect, healEffect, convertEffect;  // Different per tribe
    private int effectDrawingIdx = -1, effectTribeIdx;
    private ArrayList<Vector2d> effectPositions;
    private EFFECT effectType;  // What effect are we drawing?
    int delay = 5;
    final int nTilesExplosion = 12;
    final int nTilesEffect = 6;
    final int nTilesPierce = 3;

    enum EFFECT{
        EXPLOSION,
        SLASH,
        HEAL,
        CONVERT,
        PIERCE
    }

    /**
     * Dimensions of the window.
     */
    public static Dimension dimension;
    private static double isometricAngle = -45;

    GameView(Game game, InfoView inforView, Vector2d panTranslate)
    {
        this.game = game;
        this.board = game.getBoard().copy();
        this.infoView = inforView;
        this.panTranslate = panTranslate;

        gridSize = board.getSize();
//        int size = gridSize * CELL_SIZE;
//        int d = (int) Math.sqrt(size * size * 2);
        dimension = new Dimension(GUI_GAME_VIEW_SIZE, GUI_GAME_VIEW_SIZE);

//        backgroundImg = PLAIN.getImage(null);
        fogImg = ImageIO.GetInstance().getImage("img/fog.png");
        shineImg = getImageOrFallback("img/polytopia-public/Misc/ShineGFX.png", "img/shine3.png");
        starImg = getImageOrFallback("img/polytopia-public/Misc/star.png", "img/decorations/star.png");
        starShadow = ImageIO.GetInstance().getImage("img/decorations/starShadow.png");
        capitalImg = getImageOrFallback("img/polytopia-public/Misc/capital.png", "img/decorations/capital.png");
        capitalShadow = ImageIO.GetInstance().getImage("img/decorations/capitalShadow.png");
        cityWalls = getImageOrFallback("img/polytopia-public/Buildings/common/CityWallGFX.png", "img/terrain/walls.png");
        int expLength = nTilesExplosion * delay;
        int pierceLength = nTilesPierce * delay;
        explosionEffect = new Image[expLength];
        pierceEffect = new Image[pierceLength];
        for (int i = 0; i < nTilesExplosion; i++) {
            for (int j = 0; j < delay; j++) {
                explosionEffect[i*delay+j] = ImageIO.GetInstance().getImage("img/weapons/effects/explosion/tile" + String.format("%03d", i) + ".png");
            }
        }
        for (int i = 0; i < nTilesPierce; i++) {
            for (int j = 0; j < delay; j++) {
                pierceEffect[i*delay+j] = ImageIO.GetInstance().getImage("img/weapons/effects/pierce/tile" + String.format("%03d", i) + ".png");
            }
        }
        int nPlayers = game.getPlayers().length;
        slashEffect = new Image[nPlayers][];
        healEffect = new Image[nPlayers][];
        convertEffect = new Image[nPlayers][];
        int effLength = nTilesEffect * delay;
        for (int j = 0; j < nPlayers; j++) {
            slashEffect[j%nPlayers] = new Image[effLength];
            healEffect[j%nPlayers] = new Image[effLength];
            convertEffect[j%nPlayers] = new Image[effLength];
            for (int i = 0; i < nTilesEffect; i++) {
                for (int k = 0; k < delay; k++) {
                    slashEffect[j % nPlayers][i*delay + k] = ImageIO.GetInstance().getImage("img/weapons/effects/slash/" + j + "/tile" + String.format("%03d", i) + ".png");
                    healEffect[j % nPlayers][i*delay + k] = ImageIO.GetInstance().getImage("img/weapons/effects/heal/" + j + "/tile" + String.format("%03d", i) + ".png");
                    convertEffect[j % nPlayers][i*delay + k] = ImageIO.GetInstance().getImage("img/weapons/effects/convert/" + j + "/tile" + String.format("%03d", i) + ".png");
                }
            }
        }

        sourceTargetAnimationInfo = new ArrayList<>();
        animationSpeed = new ArrayList<>();
        actionAnimationUnitsTribe = new ArrayList<>();
        effectPositions = new ArrayList<>();
    }

    private static Image getImageOrFallback(String imagePath, String fallbackPath) {
        Image image = ImageIO.GetInstance().getImage(imagePath);
        if (image != null) {
            return image;
        }
        return ImageIO.GetInstance().getImage(fallbackPath);
    }


    public void paintComponent(Graphics gx)
    {
        super.paintComponent(gx);
        if (getWidth() > 0 && getHeight() > 0) {
            dimension = new Dimension(getWidth(), getHeight());
        }
        Graphics2D g = (Graphics2D) gx;
        paintWithGraphics(g);
    }

    private void paintWithGraphics(Graphics2D g)
    {
        if(gameState == null)
            return;

        //For a better graphics, enable this: (be aware this could bring performance issues depending on your HW & OS).
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);

        g.setColor(Color.BLACK);
        g.fillRect(0, 0, dimension.width, dimension.height);

        // Update list of actionable tiles to be highlighted (collectible resources)
        updateActionableTiles();
        paintTerrains(g);
        paintRoads(g);
        paintTerrainOverlays(g);
        paintCities(g);
        paintResourcesBuildings(g);

        int highlightX = infoView.getHighlightX();
        int highlightY = infoView.getHighlightY();

        highlightTile(g, highlightX, highlightY);

        drawCityDecorations(g);
        paintUnits(g);
        paintActionsHighlightedUnit(g, highlightX, highlightY);
        paintOtherActions(g);
        if (GUI_DRAW_EFFECTS) {
            paintEffects(g);
        }
        paintActionAnimations(g);

        g.setColor(Color.BLACK);
        //player.draw(g); //if we want to give control to the agent to paint something (for debug), start here.
    }

    private void updateActionableTiles() {
        if (actionable == null || actionable.length != gridSize || actionable[0].length != gridSize) {
            actionable = new boolean[gridSize][gridSize];
        } else {
            for (int i = 0; i < gridSize; i++) {
                java.util.Arrays.fill(actionable[i], false);
            }
        }
        HashMap<Integer, ArrayList<Action>> actions = gameState.getCityActions();
        for (Map.Entry<Integer, ArrayList<Action>> e: actions.entrySet()) {
            for (Action a: e.getValue()) {
                if (a instanceof CityAction) {
                    Vector2d pos = ((CityAction) a).getTargetPos();
                    if (pos != null) {
                        actionable[pos.x][pos.y] = a.getActionType() == Types.ACTION.RESOURCE_GATHERING;
                    }
                }
            }
        }
    }

    private void paintTerrains(Graphics2D g) {
        for(int i = 0; i < gridSize; ++i) {
            for(int j = 0; j < gridSize; ++j) {
                Types.TERRAIN t = board.getTerrainAt(i,j);
                Image toPaint;
                if (t == null || t == FOG) {
                    toPaint = fogImg;
                } else if (t == SHALLOW_WATER || t == DEEP_WATER || t == PLAIN) {
                    toPaint = getContextImg(i, j, t);
                } else {
                    toPaint = getContextImg(i, j, PLAIN);
                }
                if (t == null || t == FOG) {
                    paintFogTile(g, j, i, toPaint, panTranslate);
                } else {
                    paintTerrainTile(g, j, i, toPaint, panTranslate);
                }
            }
        }
    }

    private void paintTerrainOverlays(Graphics2D g) {
        ArrayList<int[]> overlays = new ArrayList<>();
        for(int i = 0; i < gridSize; ++i) {
            for(int j = 0; j < gridSize; ++j) {
                Types.TERRAIN t = board.getTerrainAt(i,j);
                if (t == FOREST || t == MOUNTAIN || t == VILLAGE) {
                    overlays.add(new int[]{j, i, t.getKey()});
                }
            }
        }
        overlays.sort(Comparator.comparingDouble(a -> tileCenter(a[0], a[1]).y));
        for (int[] overlay : overlays) {
            int gridX = overlay[0];
            int gridY = overlay[1];
            Types.TERRAIN terrain = Types.TERRAIN.getTypeByKey(overlay[2]);
            Image image = getContextImg(gridY, gridX, terrain);
            if (terrain == MOUNTAIN || terrain == VILLAGE) {
                paintGridImageAnchored(g, gridX, gridY, image, terrainOverlaySize(terrain), 0.5, 1.0,
                        terrainOverlayXOffset(terrain), terrainOverlayYOffset(terrain), panTranslate);
            } else {
                paintGridImage(g, gridX, gridY, image,
                        terrainOverlaySize(terrain), terrainOverlayXOffset(terrain), terrainOverlayYOffset(terrain), panTranslate);
            }
        }
    }

    private void paintCities(Graphics2D g) {
        for(int i = 0; i < gridSize; ++i) {
            for(int j = 0; j < gridSize; ++j) {
                Types.TERRAIN t = board.getTerrainAt(i,j);
                if (t == CITY) {
                    Image toPaint = t.getImage(null);
                    paintGridImage(g, j, i, toPaint, terrainOverlaySize(t), 0, -CELL_SIZE * 0.18, panTranslate);
                }
            }
        }
    }

    private static int terrainTileSize() {
        return Math.max(1, (int)Math.round(tileWidth() * 1.02));
    }

    private static int terrainOverlaySize(Types.TERRAIN terrain) {
        if (terrain == MOUNTAIN) {
            return Math.max(1, (int)Math.round(CELL_SIZE * 1.52));
        }
        if (terrain == FOREST) {
            return Math.max(1, (int)Math.round(CELL_SIZE * 1.05));
        }
        if (terrain == VILLAGE) {
            return Math.max(1, (int)Math.round(CELL_SIZE * 0.78));
        }
        return Math.max(1, (int)Math.round(CELL_SIZE * 1.05));
    }

    private static double terrainOverlayXOffset(Types.TERRAIN terrain) {
        if (terrain == VILLAGE) {
            return -CELL_SIZE * 0.02;
        }
        return 0;
    }

    private static double terrainOverlayYOffset(Types.TERRAIN terrain) {
        if (terrain == VILLAGE) {
            return CELL_SIZE * 0.20;
        }
        if (terrain == MOUNTAIN) {
            return CELL_SIZE * 0.5;
        }
        return -CELL_SIZE * 0.04;
    }

    private void paintRoads(Graphics2D g) {
        for(int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                if (isVisibleRoadTile(i, j)) {
                    boolean connected = false;
                    for (int[] direction : ROAD_DIRECTIONS) {
                        int nx = i + direction[0];
                        int ny = j + direction[1];
                        if (nx < 0 || ny < 0 || nx >= gridSize || ny >= gridSize) {
                            continue;
                        }
                        if (isRoadConnector(nx, ny)) {
                            if (!isVisibleRoadTile(nx, ny) || i < nx || (i == nx && j < ny)) {
                                drawRoadSegment(g, j, i, ny, nx);
                            }
                            connected = true;
                        }
                    }
                    if (!connected) {
                        drawRoadNode(g, j, i);
                    }
                }
            }
        }
    }

    private boolean isVisibleRoadTile(int x, int y) {
        return board.isRoad(x, y) || board.isBridge(x, y);
    }

    private boolean isRoadConnector(int x, int y) {
        if (isVisibleRoadTile(x, y)) {
            return true;
        }
        if (!board.checkTradeNetwork(x, y)) {
            return false;
        }
        Types.TERRAIN terrain = board.getTerrainAt(x, y);
        return terrain == CITY || terrain == VILLAGE || board.getBuildingAt(x, y) == Types.BUILDING.PORT;
    }

    private void drawRoadSegment(Graphics2D g, int gridX1, int gridY1, int gridX2, int gridY2) {
        Point2D.Double p1 = tileCenter(gridX1, gridY1);
        Point2D.Double p2 = tileCenter(gridX2, gridY2);
        int x1 = (int)Math.round(p1.x + panTranslate.x);
        int y1 = (int)Math.round(p1.y + panTranslate.y);
        int x2 = (int)Math.round(p2.x + panTranslate.x);
        int y2 = (int)Math.round(p2.y + panTranslate.y);

        Stroke oldStroke = g.getStroke();
        Color oldColor = g.getColor();
        g.setStroke(new BasicStroke(Math.max(3.0f, CELL_SIZE / 10.0f), BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g.setColor(new Color(102, 80, 49, 120));
        g.drawLine(x1, y1 + Math.max(1, CELL_SIZE / 34), x2, y2 + Math.max(1, CELL_SIZE / 34));
        g.setStroke(new BasicStroke(Math.max(2.0f, CELL_SIZE / 13.0f), BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g.setColor(new Color(184, 133, 72, 235));
        g.drawLine(x1, y1, x2, y2);
        g.setStroke(new BasicStroke(Math.max(1.0f, CELL_SIZE / 32.0f), BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        g.setColor(new Color(221, 174, 102, 170));
        g.drawLine(x1, y1 - Math.max(1, CELL_SIZE / 50), x2, y2 - Math.max(1, CELL_SIZE / 50));
        g.setStroke(oldStroke);
        g.setColor(oldColor);
    }

    private void drawRoadNode(Graphics2D g, int gridX, int gridY) {
        Point2D.Double center = tileCenter(gridX, gridY);
        int diameter = Math.max(4, CELL_SIZE / 7);
        int x = (int)Math.round(center.x + panTranslate.x - diameter / 2.0);
        int y = (int)Math.round(center.y + panTranslate.y - diameter / 2.0);
        Color oldColor = g.getColor();
        g.setColor(new Color(184, 133, 72, 225));
        g.fillOval(x, y, diameter, diameter);
        g.setColor(oldColor);
    }

    private void paintResourcesBuildings(Graphics2D g) {
        for(int i = 0; i < gridSize; ++i) {
            for(int j = 0; j < gridSize; ++j) {
                Types.TERRAIN t = board.getTerrainAt(i,j);
                Types.RESOURCE r = board.getResourceAt(i,j);
                if (actionable[i][j]) paintGridImage(g, j, i, shineImg, CELL_SIZE, 0, panTranslate);
                Image resourceImg = (r == null) ? null : r.getImage(t);
                if (r == Types.RESOURCE.CROPS) {
                    paintTileCoverImage(g, j, i, resourceImg, 0.68, CELL_SIZE * 0.01, panTranslate);
                } else {
                    int imgSize = resourceSize(r);
                    paintGridImage(g, j, i, resourceImg, imgSize, resourceYOffset(r), panTranslate);
                }

                Types.BUILDING b = board.getBuildingAt(i,j);
                int imgSize = buildingSize(b);
                paintGridImage(g, j, i, (b == null) ? null : b.getImage(), imgSize, buildingYOffset(b), panTranslate);
            }
        }
    }

    private static int resourceSize(Types.RESOURCE resource) {
        if (resource == null) {
            return Math.max(1, (int)Math.round(CELL_SIZE * 0.65));
        }
        switch (resource) {
            case ORE:
                return Math.max(1, (int)Math.round(CELL_SIZE * 0.50));
            case ANIMAL:
                return Math.max(1, (int)Math.round(CELL_SIZE * 0.38));
            case LIGHTHOUSE:
                return Math.max(1, (int)Math.round(CELL_SIZE * 0.78));
            case RUINS:
                return Math.max(1, (int)Math.round(CELL_SIZE * 0.85));
            case STARFISH:
                return Math.max(1, (int)Math.round(CELL_SIZE * 0.52));
            default:
                return Math.max(1, (int)Math.round(CELL_SIZE * 0.62));
        }
    }

    private static double resourceYOffset(Types.RESOURCE resource) {
        if (resource == null) {
            return 0;
        }
        switch (resource) {
            case ANIMAL:
                return -CELL_SIZE * 0.12;
            case FISH:
                return -CELL_SIZE * 0.03;
            case STARFISH:
                return -CELL_SIZE * 0.03;
            case FRUIT:
            case CROPS:
                return -CELL_SIZE * 0.18;
            case ORE:
                return -CELL_SIZE * 0.16;
            case LIGHTHOUSE:
                return -CELL_SIZE * 0.14;
            case RUINS:
                return -CELL_SIZE * 0.10;
            default:
                return -CELL_SIZE * 0.12;
        }
    }

    private static int buildingSize(Types.BUILDING building) {
        if (building == null) {
            return Math.max(1, CELL_SIZE);
        }
        if (building == Types.BUILDING.MARKET) {
            return Math.max(1, (int)Math.round(CELL_SIZE * 0.95));
        }
        if (building.isTemple()) {
            return Math.max(1, (int)Math.round(CELL_SIZE * 0.90));
        }
        return Math.max(1, (int)Math.round(CELL_SIZE * 0.95));
    }

    private static double buildingYOffset(Types.BUILDING building) {
        if (building == null) {
            return 0;
        }
        if (building == Types.BUILDING.PORT) {
            return -CELL_SIZE * 0.06;
        }
        return -CELL_SIZE * 0.10;
    }

    private void highlightTile(Graphics2D g, int highlightX, int highlightY) {
        if (highlightX != -1) {
            Stroke oldStroke = g.getStroke();
            g.setColor(Color.BLUE);
            g.setStroke(new BasicStroke(3));

            drawTileOutline(g, highlightX, highlightY, panTranslate);
            g.setStroke(oldStroke);
            g.setColor(Color.BLACK);
        }
    }

    private void paintUnits(Graphics2D g) {
        for(int i = 0; i < gridSize; ++i) {
            for(int j = 0; j < gridSize; ++j) {
                Unit u = board.getUnitAt(i,j);
                if (u != null) {
                    int imgSize = Math.max(1, (int) (CELL_SIZE * 0.60));
                    if (u instanceof SuperUnit || u instanceof Catapult) imgSize = Math.max(1, (int)(CELL_SIZE * 0.84));

                    Point2D.Double center = tileCenter(j, i);
                    int x = (int)Math.round(center.x);
                    int y = (int)Math.round(center.y + CELL_SIZE * 0.18);

                    ArrayList<Action> possibleActions = gameState.getUnitActions(u);
                    boolean exhausted = (possibleActions == null || possibleActions.size() == 0);

                    Tribe t = gameState.getTribe(u.getTribeId());
                    int imageTribeId = t.getType().getKey();

                    paintUnit(g, x, y, u.getType(), imgSize, imageTribeId, exhausted);

                    if (u.getType().isWaterUnit()) {
                        // Paint base land unit for water units
                        int size = imgSize/2;
                        paintUnit(g, x + CELL_SIZE/6, y + CELL_SIZE/8, board.getBaseLandUnit(u), size, imageTribeId, exhausted);
                    }

                    Font f = g.getFont();
                    g.setFont(new Font(f.getFontName(), Font.PLAIN, CELL_SIZE/5));
                    g.setColor(Color.black);
                    g.drawString(u.getCurrentHP() + "/" + u.getMaxHP(),
                            x - CELL_SIZE/6 + panTranslate.x, y - CELL_SIZE/2 + panTranslate.y);
                    g.setFont(f);
                }
            }
        }
    }

    private void paintUnit(Graphics2D g, int x, int y, Types.UNIT type, int imgSize, int tribeId, boolean exhausted) {
        Image unitImg = type.getImage(tribeId);
        Point2D.Double anchor = getUnitAnchor(unitImg);
        paintImageTrimmedAnchored(g, x, y, unitImg, imgSize, anchor.x, anchor.y, panTranslate, exhausted ? 0.90f : 1.0f);
    }

    private void paintActionsHighlightedUnit(Graphics2D g, int highlightX, int highlightY) {
        if (infoView.highlightInGridBounds()) {
            Unit u = board.getUnitAt(highlightY, highlightX);
            if (u != null && !infoView.clickedTwice()) {
                ArrayList<Action> possibleActions = gameState.getUnitActions(u);
                if (possibleActions != null && possibleActions.size() > 0) {
                    for (Action a : possibleActions) {
                        if (!(a.getActionType() == Types.ACTION.EXAMINE || a.getActionType() == Types.ACTION.CAPTURE)) {
                            Image actionImg = Types.ACTION.getImage(a);

                            if (actionImg != null) {
                                Vector2d pos = GUI.getActionPosition(gameState, a);

                                if (pos != null) {
                                    if (a.getActionType() == Types.ACTION.MOVE) {
                                        paintImageRotated(g, pos.y * CELL_SIZE, pos.x * CELL_SIZE, actionImg, CELL_SIZE, panTranslate);
                                    } else {
                                        Vector2d rotated = rotatePoint(pos.y, pos.x);
                                        paintImage(g, rotated.x, rotated.y - CELL_SIZE/2, actionImg, CELL_SIZE, panTranslate);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private void paintOtherActions(Graphics2D g) {
        HashMap<Integer, ArrayList<Action>> actions = gameState.getUnitActions();
        for (Map.Entry<Integer, ArrayList<Action>> e: actions.entrySet()) {
            for (Action a : e.getValue()) {
                if (a.getActionType() == EXAMINE || a.getActionType() == CAPTURE) {
                    Image actionImg = Types.ACTION.getImage(a);
                    if (actionImg != null) {
                        Vector2d pos = GUI.getActionPosition(gameState, a);

                        if (pos != null) {
                            Vector2d rotated = rotatePoint(pos.y, pos.x);
                            int imgSize = (int)(CELL_SIZE*0.5);
                            paintImage(g, rotated.x + CELL_SIZE, rotated.y - imgSize/2,
                                    actionImg, imgSize, panTranslate);
                        }
                    }
                }
            }
        }
    }

    // *****************************************************************************************************************

    private static void paintTerrainTile(Graphics2D g, int gridX, int gridY, Image img, Vector2d panTranslate) {
        if (img == null) {
            return;
        }
        Point2D.Double center = tileCenter(gridX, gridY);
        int width = terrainTileSize();
        int height = Math.max(1, (int)Math.round(width * TILE_HEIGHT_RATIO));
        int drawX = (int)Math.round(center.x - width / 2.0);
        int drawY = (int)Math.round(center.y - height / 2.0);
        g.drawImage(img, drawX + panTranslate.x, drawY + panTranslate.y, width, height, null);
    }

    private static void paintTileCoverImage(Graphics2D g, int gridX, int gridY, Image img, double widthScale,
                                            double yOffset, Vector2d panTranslate) {
        if (img == null) {
            return;
        }
        Point2D.Double center = tileCenter(gridX, gridY);
        int width = Math.max(1, (int)Math.round(terrainTileSize() * widthScale));
        int height = Math.max(1, (int)Math.round(width * TILE_HEIGHT_RATIO));
        int drawX = (int)Math.round(center.x - width / 2.0);
        int drawY = (int)Math.round(center.y - height / 2.0 + yOffset);
        g.drawImage(img, drawX + panTranslate.x, drawY + panTranslate.y, width, height, null);
    }

    private static void paintFogTile(Graphics2D g, int gridX, int gridY, Image img, Vector2d panTranslate) {
        if (img == null) {
            return;
        }
        Polygon tile = tilePolygon(gridX, gridY, panTranslate);
        Rectangle bounds = tile.getBounds();
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setClip(tile);
        g2.clipRect(0, 0, dimension.width, dimension.height);
        g2.drawImage(img, bounds.x, bounds.y, bounds.width, bounds.height, null);
        g2.dispose();
    }

    private static void paintGridImage(Graphics2D g, int gridX, int gridY, Image img, int targetSize, Vector2d panTranslate) {
        paintGridImage(g, gridX, gridY, img, targetSize, 0, panTranslate);
    }

    private static void paintGridImage(Graphics2D g, int gridX, int gridY, Image img, int targetSize,
                                       double yOffset, Vector2d panTranslate) {
        paintGridImage(g, gridX, gridY, img, targetSize, 0, yOffset, panTranslate);
    }

    private static void paintGridImage(Graphics2D g, int gridX, int gridY, Image img, int targetSize,
                                       double xOffset, double yOffset, Vector2d panTranslate) {
        if (img == null) {
            return;
        }
        Point2D.Double center = tileCenter(gridX, gridY);
        paintImageTrimmedCentered(g, (int)Math.round(center.x + xOffset), (int)Math.round(center.y + yOffset),
                img, targetSize, panTranslate, 1.0f);
    }

    private static void paintGridImageAnchored(Graphics2D g, int gridX, int gridY, Image img, int targetSize,
                                               double anchorX, double anchorY, double xOffset, double yOffset,
                                               Vector2d panTranslate) {
        if (img == null) {
            return;
        }
        Point2D.Double center = tileCenter(gridX, gridY);
        paintImageTrimmedAnchored(g, (int)Math.round(center.x + xOffset), (int)Math.round(center.y + yOffset),
                img, targetSize, anchorX, anchorY, panTranslate, 1.0f);
    }

    private static void paintImageRotated(Graphics2D gphx, int x, int y, Image img, int imgSize, Vector2d panTranslate) {
        if (img != null) {
            int w = img.getWidth(null);
            int h = img.getHeight(null);
            float scaleX = (float)imgSize/w;
            float scaleY = (float)imgSize/h;
            Point2D.Double rotated = rotatePointExact(1.0 * x / CELL_SIZE, 1.0 * y / CELL_SIZE);
            int drawX = (int)Math.round(rotated.x + CELL_SIZE / 2.0 - imgSize / 2.0);
            int drawY = (int)Math.round(rotated.y + CELL_SIZE / 2.0 - imgSize / 2.0);
            gphx.drawImage(img, drawX + panTranslate.x, drawY + panTranslate.y,
                    (int) (w*scaleX), (int) (h*scaleY), null);
        }
    }

    private static void paintImageRotated(Graphics2D gphx, int x, int y, Image img, int imgSize, Vector2d panTranslate,
                                          double angle, int xAnchor, int yAnchor) {
        if (img != null) {
            int w = img.getWidth(null);
            int h = img.getHeight(null);
            float scaleX = (float)imgSize/w;
            float scaleY = (float)imgSize/h;

            Graphics2D g2 = (Graphics2D)gphx.create();
            g2.translate(panTranslate.x, panTranslate.y);
            g2.rotate(angle, xAnchor, yAnchor);
            g2.drawImage(img, x, y, (int) (w*scaleX), (int) (h*scaleY), null);
            g2.dispose();
        }
    }

    private static void paintImage(Graphics2D gphx, int x, int y, Image img, int imgSize, Vector2d panTranslate)
    {
        paintImage(gphx, x, y, img, imgSize, panTranslate, 1.0f);
    }

    private static void paintImage(Graphics2D gphx, int x, int y, Image img, int imgSize, Vector2d panTranslate,
                                   float alpha)
    {
        if (img != null) {
            int w = img.getWidth(null);
            int h = img.getHeight(null);
            float scaleX = (float)imgSize/w;
            float scaleY = (float)imgSize/h;
            Composite oldComposite = gphx.getComposite();
            if (alpha < 1.0f) {
                gphx.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
            }
            gphx.drawImage(img, x + panTranslate.x, y + panTranslate.y,
                    (int) (w*scaleX), (int) (h*scaleY), null);
            if (alpha < 1.0f) {
                gphx.setComposite(oldComposite);
            }
        }
    }

    private static void paintImageTrimmedCentered(Graphics2D gphx, int centerX, int centerY, Image img, int targetSize,
                                                  Vector2d panTranslate, float alpha) {
        paintImageTrimmedAnchored(gphx, centerX, centerY, img, targetSize, 0.5, 0.5, panTranslate, alpha);
    }

    private static void paintImageTrimmedAnchored(Graphics2D gphx, int anchorX, int anchorY, Image img, int targetSize,
                                                  double boundsAnchorX, double boundsAnchorY,
                                                  Vector2d panTranslate, float alpha) {
        if (img == null) {
            return;
        }
        Rectangle bounds = getOpaqueBounds(img);
        int w = img.getWidth(null);
        int h = img.getHeight(null);
        double maxDim = Math.max(bounds.width, bounds.height);
        if (maxDim <= 0) {
            return;
        }
        double scale = targetSize / maxDim;
        int drawW = Math.max(1, (int)Math.round(w * scale));
        int drawH = Math.max(1, (int)Math.round(h * scale));
        int scaledAnchorX = (int)Math.round((bounds.x + bounds.width * boundsAnchorX) * scale);
        int scaledAnchorY = (int)Math.round((bounds.y + bounds.height * boundsAnchorY) * scale);
        int drawX = anchorX - scaledAnchorX;
        int drawY = anchorY - scaledAnchorY;

        Composite oldComposite = gphx.getComposite();
        if (alpha < 1.0f) {
            gphx.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
        }
        gphx.drawImage(img, drawX + panTranslate.x, drawY + panTranslate.y, drawW, drawH, null);
        if (alpha < 1.0f) {
            gphx.setComposite(oldComposite);
        }
    }

    private static Rectangle getOpaqueBounds(Image img) {
        Rectangle cached = opaqueBoundsCache.get(img);
        if (cached != null) {
            return cached;
        }
        int w = img.getWidth(null);
        int h = img.getHeight(null);
        if (w <= 0 || h <= 0) {
            Rectangle fallback = new Rectangle(0, 0, 1, 1);
            opaqueBoundsCache.put(img, fallback);
            return fallback;
        }
        BufferedImage buffered = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = buffered.createGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();

        int minX = w;
        int minY = h;
        int maxX = -1;
        int maxY = -1;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int alpha = (buffered.getRGB(x, y) >>> 24) & 0xff;
                if (alpha > 8) {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }

        Rectangle bounds = maxX >= minX && maxY >= minY
                ? new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1)
                : new Rectangle(0, 0, w, h);
        opaqueBoundsCache.put(img, bounds);
        return bounds;
    }

    private static Point2D.Double getUnitAnchor(Image img) {
        if (img == null) {
            return new Point2D.Double(0.5, 1.0);
        }
        Point2D.Double cached = unitAnchorCache.get(img);
        if (cached != null) {
            return cached;
        }

        Rectangle bounds = getOpaqueBounds(img);
        int w = img.getWidth(null);
        int h = img.getHeight(null);
        if (w <= 0 || h <= 0 || bounds.width <= 0 || bounds.height <= 0) {
            Point2D.Double fallback = new Point2D.Double(0.5, 1.0);
            unitAnchorCache.put(img, fallback);
            return fallback;
        }

        BufferedImage buffered = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = buffered.createGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();

        int startY = bounds.y + Math.max(0, (int)Math.round(bounds.height * 0.68));
        long weightedX = 0;
        long weight = 0;
        int bottomY = bounds.y + bounds.height - 1;
        for (int y = startY; y <= bottomY; y++) {
            int rowWeight = Math.max(1, y - startY + 1);
            for (int x = bounds.x; x < bounds.x + bounds.width; x++) {
                int alpha = (buffered.getRGB(x, y) >>> 24) & 0xff;
                if (alpha > 8) {
                    weightedX += (long)x * rowWeight;
                    weight += rowWeight;
                }
            }
        }

        double anchorX = weight > 0
                ? ((weightedX / (double)weight) - bounds.x) / bounds.width
                : 0.5;
        Point2D.Double anchor = new Point2D.Double(Math.max(0.0, Math.min(1.0, anchorX)), 1.0);
        unitAnchorCache.put(img, anchor);
        return anchor;
    }

    private static void drawTileOutline(Graphics2D g, int gridX, int gridY, Vector2d panTranslate) {
        Polygon polygon = tilePolygon(gridX, gridY, panTranslate);
        g.drawPolygon(polygon);
    }

    private static Polygon tilePolygon(int gridX, int gridY, Vector2d panTranslate) {
        Point2D.Double p0 = rotatePointExact(gridX, gridY);
        Point2D.Double p1 = rotatePointExact(gridX + 1.0, gridY);
        Point2D.Double p2 = rotatePointExact(gridX + 1.0, gridY + 1.0);
        Point2D.Double p3 = rotatePointExact(gridX, gridY + 1.0);
        int[] xs = {
                (int)Math.round(p0.x + panTranslate.x),
                (int)Math.round(p1.x + panTranslate.x),
                (int)Math.round(p2.x + panTranslate.x),
                (int)Math.round(p3.x + panTranslate.x)
        };
        int[] ys = {
                (int)Math.round(p0.y + panTranslate.y),
                (int)Math.round(p1.y + panTranslate.y),
                (int)Math.round(p2.y + panTranslate.y),
                (int)Math.round(p3.y + panTranslate.y)
        };
        return new Polygon(xs, ys, 4);
    }

    private static double tileWidth() {
        return CELL_SIZE * TILE_WIDTH_SCALE;
    }

    private static double tileHeight() {
        return tileWidth() * TILE_HEIGHT_RATIO;
    }

    private static Point2D.Double tileCenter(double gridX, double gridY) {
        return rotatePointExact(gridX + 0.5, gridY + 0.5);
    }

    /**
     * Expects coordinates in grid, translates to screen coordinates.
     */
    public static Vector2d rotatePoint(double x, double y) {
        Point2D.Double point = rotatePointExact(x, y);
        return new Vector2d((int)point.x, (int)point.y);
    }

    public static Point2D.Double rotatePointExact(double x, double y) {
        double x2 = (x + y) * tileWidth() / 2.0;
        double y2 = (y - x) * tileHeight() / 2.0;
        y2 += dimension.width/2.0;
        return new Point2D.Double(x2, y2);
    }

    /**
     * Expects screen coordinates, returns coordinates in grid.
     */
    public static Vector2d rotatePointReverse(double x, double y) {
        Point2D.Double point = rotatePointReverseExact(x, y);
        return new Vector2d((int)point.x, (int)point.y);
    }

    public static Point2D.Double rotatePointReverseExact(double x, double y) {
        y -= GameView.dimension.width/2.0;
        double sum = 2.0 * x / tileWidth();
        double diff = 2.0 * y / tileHeight();
        double gridX = (sum - diff) / 2.0;
        double gridY = (sum + diff) / 2.0;
        return new Point2D.Double(gridX, gridY);
    }

    private void drawCityDecorations(Graphics2D g) {
        for(int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                Types.TERRAIN terrainAt = board.getTerrainAt(i, j);
                if (terrainAt == CITY) {
                    int d = (int)Math.sqrt(CELL_SIZE*CELL_SIZE*2);
                    int fontSize = CELL_SIZE/3;
                    Font textFont = new Font(getFont().getName(), Font.PLAIN, fontSize);
                    g.setFont(textFont);

                    int cityID = board.getCityIdAt(i,j);
                    City c = (City) board.getActor(cityID);

                    if (c != null) {
                        int cityCapacity = c.getLevel() + 1;
                        int progress = c.getPopulation();
                        int units = c.getUnitsID().size();
                        Tribe tr = gameState.getTribe(c.getTribeId());
                        Color col = Types.TRIBE.values()[tr.getType().getKey()].getColorDark();
                        Color colTransparent = new Color(col.getRed(), col.getGreen(), col.getBlue(), 170);

                        // Draw city walls
                        if (c.hasWalls()) {
                            Point2D.Double cityCenter = tileCenter(j, i);
                            paintImageTrimmedAnchored(g,
                                    (int)Math.round(cityCenter.x),
                                    (int)Math.round(cityCenter.y + CELL_SIZE * 0.23),
                                    cityWalls,
                                    Math.max(1, (int)Math.round(CELL_SIZE * 1.10)),
                                    0.5,
                                    0.70,
                                    panTranslate,
                                    1.0f);
                        }

                        // Draw city border
                        LinkedList<Vector2d> tiles = board.getCityTiles(cityID);
                        int nCityTiles = tiles.size();
                        int nNeighbours = 4;
                        boolean[][] tileNeighbours = new boolean[nCityTiles][nNeighbours];  // order: left, right, up, down
                        Pair<Vector2d, Vector2d>[] lines = new Pair[]{new Pair<>(new Vector2d(0, 0), new Vector2d(0, 1)),
                                new Pair<>(new Vector2d(1, 0), new Vector2d(1, 1)),
                                new Pair<>(new Vector2d(0, 0), new Vector2d(1, 0)),
                                new Pair<>(new Vector2d(0, 1), new Vector2d(1, 1))};
                        for (int k = 0; k < nCityTiles-1; k++) {
                            Vector2d t1 = tiles.get(k);
                            for (int p = k+1; p < nCityTiles; p++) {
                                Vector2d t2 = tiles.get(p);
                                if (t1.equals(t2)) continue;
                                if (t1.x - t2.x == 1 && t1.y == t2.y) {  // t1 to the right of t2
                                    tileNeighbours[k][0] = true;
                                    tileNeighbours[p][1] = true;
                                } else if (t1.x - t2.x == -1 && t1.y == t2.y) { // t1 to the left of t2
                                    tileNeighbours[k][1] = true;
                                    tileNeighbours[p][0] = true;
                                } else if (t1.x == t2.x && t1.y - t2.y == 1) {  // t1 underneath t2
                                    tileNeighbours[k][2] = true;
                                    tileNeighbours[p][3] = true;
                                } else if (t1.x == t2.x && t1.y - t2.y == -1) {  // t1 above t2
                                    tileNeighbours[k][3] = true;
                                    tileNeighbours[p][2] = true;
                                }
                            }
                        }
                        g.setColor(new Color(col.getRed(), col.getGreen(), col.getBlue(), 145));
                        Stroke oldStroke = g.getStroke();
                        g.setStroke(new BasicStroke(Math.max(2.0f, CELL_SIZE / 18.0f),
                                BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 0,
                                new float[]{Math.max(5.0f, CELL_SIZE / 8.0f)}, 0));
                        // Draw lines for tiles that don't have a neighbour on a side
                        for (int t = 0; t < nCityTiles; t++) {
                            Vector2d tile = tiles.get(t);
                            for (int n = 0; n < nNeighbours; n++) {
                                if (!tileNeighbours[t][n]) {
                                    // draw line on this side
                                    Vector2d from = rotatePoint(lines[n].getFirst().y + tile.y, lines[n].getFirst().x + tile.x);
                                    Vector2d to = rotatePoint(lines[n].getSecond().y + tile.y, lines[n].getSecond().x + tile.x);
                                    g.drawLine(from.x + panTranslate.x, from.y + panTranslate.y,
                                            to.x + panTranslate.x, to.y + panTranslate.y);
                                }
                            }
                        }
                        g.setStroke(oldStroke);

                        // Draw capital img + city name/ID + number of stars
                        String cityName = "" + cityID;
                        String production = "" + c.getProduction();

                        int sections = 2;
                        if (c.isCapital()) {
                            sections = 3;
                        }
                        int h = d / 4;
                        double nameWidth = GUI_CITY_TAG_WIDTH + sections * h;
                        Vector2d namePos = rotatePoint(j, i);
                        Rectangle nameRect = new Rectangle((int) Math.round(namePos.x + d / 2.0 - nameWidth / 2.0),
                                (int) (namePos.y + d / 2.0 - h), (int) nameWidth, h);
                        g.setColor(colTransparent);
                        g.fillRect(nameRect.x + panTranslate.x, nameRect.y + panTranslate.y, nameRect.width, nameRect.height);
                        g.setColor(Color.WHITE);
                        FontMetrics metrics = g.getFontMetrics();
                        int leftReserved = c.isCapital() ? h : 0;
                        int rightReserved = Math.max(h * 2, (int) Math.round(nameRect.width * 0.34));
                        int labelLeft = nameRect.x + leftReserved;
                        int labelWidth = Math.max(1, nameRect.width - leftReserved - rightReserved);
                        drawCenteredString(g, cityName, labelLeft, labelWidth,
                                (int) (nameRect.y + h * 1.1 - fontSize / 4.0), metrics);

                        // Draw number of stars
                        paintImage(g, (int) (nameRect.x + nameRect.width * (0.35 + (sections-2)*0.2) + SHADOW_OFFSET),
                                nameRect.y + SHADOW_OFFSET, starShadow, h, panTranslate);
                        paintImage(g, (int) (nameRect.x + nameRect.width * (0.35 + (sections-2)*0.2)), nameRect.y, starImg, h, panTranslate);
                        drawStringShadow(g, production, (int) (nameRect.x + nameRect.width - fontSize * 0.75),
                                (int) (nameRect.y + h * 1.1 - fontSize / 4.0));
                        g.setColor(Color.WHITE);
                        g.drawString(production, (int) (nameRect.x + nameRect.width - fontSize * 0.75 + panTranslate.x),
                                (int) (nameRect.y + h * 1.1 - fontSize / 4.0 + panTranslate.y));

                        // Draw capital sign
                        if (c.isCapital()) {
                            paintImage(g, nameRect.x + SHADOW_OFFSET, nameRect.y + SHADOW_OFFSET, capitalShadow, h, panTranslate);
                            paintImage(g, nameRect.x, nameRect.y, capitalImg, h, panTranslate);
                        }

                        // Draw level
                        h /= 2;
                        int sectionWidth = h;
                        int w = cityCapacity * sectionWidth;
                        Rectangle bgRect = new Rectangle(nameRect.x + nameRect.width / 2 - w / 2, nameRect.y + nameRect.height, w, h);
                        drawRoundRectShadowHighlight(g, bgRect);
                        g.setColor(Color.WHITE);
                        g.fillRoundRect(bgRect.x + panTranslate.x, bgRect.y + panTranslate.y,
                                bgRect.width, bgRect.height, ROUND_RECT_ARC, ROUND_RECT_ARC);

                        // Draw population/progress
                        if (progress >= 0) {
                            g.setColor(progressColor);
                        } else {
                            g.setColor(negativeColor);
                        }
                        int pw = Math.abs(progress) * sectionWidth;
                        Rectangle pgRect = new Rectangle(bgRect.x, bgRect.y, pw, bgRect.height);
                        g.fillRoundRect(pgRect.x + panTranslate.x, pgRect.y + +panTranslate.y,
                                pgRect.width, pgRect.height, ROUND_RECT_ARC, ROUND_RECT_ARC);

                        // Draw unit counts
                        g.setColor(Color.black);
                        int radius = h / 2;
                        int unitHeight = bgRect.y + h / 2 - radius / 2;
                        for (int u = 0; u < units; u++) {
                            g.fillOval(bgRect.x + sectionWidth * u + sectionWidth / 2 - radius / 2 + panTranslate.x,
                                    unitHeight + panTranslate.y, radius, radius);
                        }

                        // Draw section separations
                        for (int l = 0; l < cityCapacity - 1; l++) {
                            int lx = bgRect.x + sectionWidth * (l + 1);
                            g.drawLine(lx + panTranslate.x, bgRect.y + panTranslate.y,
                                    lx + panTranslate.x, bgRect.y + bgRect.height + panTranslate.y);
                        }
                    }
                }
            }
        }
    }

    private void drawRoundRectShadowHighlight(Graphics2D g, Rectangle rect) {
        g.setColor(new Color(0, 0, 0, 122));
        g.fillRoundRect(rect.x + SHADOW_OFFSET + panTranslate.x,
                rect.y + SHADOW_OFFSET + panTranslate.y, rect.width, rect.height,
                ROUND_RECT_ARC, ROUND_RECT_ARC);
        g.setColor(new Color(255, 255, 255, 122));
        g.fillRoundRect(rect.x - SHADOW_OFFSET + panTranslate.x,
                rect.y - SHADOW_OFFSET + panTranslate.y, rect.width, rect.height,
                ROUND_RECT_ARC, ROUND_RECT_ARC);
    }

    private void drawStringShadow (Graphics2D g, String s, int x, int y) {
        g.setColor(new Color(0, 0, 0, 122));
        g.drawString(s, x+SHADOW_OFFSET + panTranslate.x, y+SHADOW_OFFSET + panTranslate.y);
    }

    private void drawCenteredString(Graphics2D g, String text, int left, int width, int baselineY, FontMetrics metrics) {
        int textWidth = metrics.stringWidth(text);
        int drawX = left + Math.max(0, (width - textWidth) / 2);
        g.drawString(text, drawX + panTranslate.x, baselineY + panTranslate.y);
    }


    /**
     * Paints the board
     * @param gs current game state
     */
    void refresh(GameState gs)
    {
        //The tribe Id of which the turn gs at this point
        //int gameTurn = 0;// gs.getTick() % gs.getTribes().length;
        gameState = gs; //.copy(gameTurn);
        board = gameState.getBoard();
    }

    void updatePan(Vector2d panTranslate) {
        this.panTranslate = new Vector2d(this.panTranslate.x + panTranslate.x,
                this.panTranslate.y + panTranslate.y);
    }

    void setPanToTribe(GameState gs) {
        // Focus on capital of tribe
        Tribe t = gs.getTribe(gs.getActiveTribeID());
        int capitalID = t.getCapitalID();
        Actor a = gs.getActor(capitalID);
        Vector2d pos = a.getPosition();

        // Get position in screen coordinates, and set pan to the negative difference to center
        Vector2d screenPoint = rotatePoint(pos.y, pos.x);
        panTranslate = new Vector2d(-screenPoint.x - CELL_SIZE/2 + dimension.width/2,
                -screenPoint.y - CELL_SIZE/2 + dimension.height/2);
    }

    public Vector2d getPanTranslate() {
        return panTranslate;
    }

    /**
     * Gets the dimensions of the window.
     * @return the dimensions of the window.
     */
    public Dimension getPreferredSize() {
        return dimension;
    }


    private Image getContextImg(int i, int j, Types.TERRAIN t) {
        Image toPaint;
        boolean cornerUL = (i == 0 && j == 0);
        boolean cornerDR = (i == gridSize - 1 && j == gridSize -1);
        Types.TERRAIN diagUR = null;
        if (i > 0 && j < gridSize - 1) diagUR = board.getTerrainAt(i - 1, j + 1);

        if (t == DEEP_WATER || t == SHALLOW_WATER) {
            // If this is the last tile on the row
            boolean down = (i == gridSize - 1);
            // If the tile above is not water
            boolean top = (i > 0 && board.getTerrainAt(i - 1, j) != SHALLOW_WATER && board.getTerrainAt(i - 1, j) != DEEP_WATER);
            // If first tile on column
            boolean left = (j == 0);
            // If the tile to the right is not water
            boolean right = (j < gridSize - 1 && board.getTerrainAt(i, j + 1) != SHALLOW_WATER && board.getTerrainAt(i, j + 1) != DEEP_WATER);
            if (down) {
                if (left) {
                    toPaint = t.getImage("down-left");
                } else if (right) {
                    if (top) {
                        toPaint = t.getImage("top-down-right");
                    } else {
                        if (t == SHALLOW_WATER && (diagUR == SHALLOW_WATER || diagUR == DEEP_WATER)) {
                            toPaint = t.getImage("down-right-ur");
                        } else {
                            toPaint = t.getImage("down-right");
                        }
                    }
                } else if (top) {
                    if (t == SHALLOW_WATER && (diagUR == SHALLOW_WATER || diagUR == DEEP_WATER)) {
                        toPaint = t.getImage("top-down-ur");
                    } else {
                        toPaint = t.getImage("top-down");
                    }
                } else {
                    toPaint = t.getImage("down");
                }
            } else if (top) {
                if (t == SHALLOW_WATER && (j == gridSize-1 || diagUR == SHALLOW_WATER || diagUR == DEEP_WATER)) {
                    if (left) {
                        toPaint = t.getImage("top-left-ur");
                    } else {
                        toPaint = t.getImage("top-ur");
                    }
                } else {
                    if (right) {
                        if (left) {
                            toPaint = t.getImage("top-left-right");
                        } else {
                            toPaint = t.getImage("top-right");
                        }
                    } else if (left) {
                        toPaint = t.getImage("top-left");
                    } else {
                        toPaint = t.getImage("top");
                    }
                }
            } else if (right) {
                if (i == 0 || j < gridSize - 1 && t == SHALLOW_WATER && (diagUR == SHALLOW_WATER || diagUR == DEEP_WATER)) {
                    toPaint = t.getImage("right-ur");
                } else {
                    toPaint = t.getImage("right");
                }
            } else if (left) {
                toPaint = t.getImage("left");
            } else {
                toPaint = t.getImage(null);
            }
        } else {
            // If this is the last tile on the row, or the tile underneath is water, get the '-down' img
            boolean down = (i == gridSize - 1 || board.getTerrainAt(i+1, j) == SHALLOW_WATER);
            // If the same is true for the column instead, this is a '-left' img
            boolean left = (j == 0 || board.getTerrainAt(i, j-1) == SHALLOW_WATER);

            if (down) {
                if (j == gridSize - 1) {
                    if (cornerDR) {
                        toPaint = t.getImage("corner-dr");
                    } else {
                        toPaint = t.getImage("down-dr");
                    }
                } else {
                    if (left) {
                        if (j == 0) {
                            if (i < gridSize - 1 && (board.getTerrainAt(i, j + 1) == SHALLOW_WATER ||
                                            board.getTerrainAt(i, j + 1) == DEEP_WATER)) {
                                toPaint = t.getImage("down-left-el-dr");
                            } else {
                                toPaint = t.getImage("down-left-el");
                            }
                        } else if (i == gridSize - 1) {
                            if (board.getTerrainAt(i - 1, j) == SHALLOW_WATER
                                    || board.getTerrainAt(i - 1, j) == DEEP_WATER) {
                                toPaint = t.getImage("down-left-ed-ul");
                            } else {
                                toPaint = t.getImage("down-left-ed");
                            }
                        } else {
                            toPaint = t.getImage("down-left");
                        }
                    } else {
                        if (j < gridSize - 1 && i < gridSize - 1 &&
                                (board.getTerrainAt(i, j + 1) == SHALLOW_WATER ||
                                board.getTerrainAt(i, j + 1) == DEEP_WATER)) {
                            toPaint = t.getImage("down-dr");
                        } else {
                            toPaint = t.getImage("down");
                        }
                    }
                }
            } else if (left) {
                if (i == 0 || board.getTerrainAt(i - 1, j) == SHALLOW_WATER ||
                        board.getTerrainAt(i - 1, j) == DEEP_WATER) {
                    if (cornerUL) {
                        toPaint = t.getImage("corner-ul");
                    } else {
                        if (j == 0) {
                            toPaint = t.getImage("left");
                        } else {
                            toPaint = t.getImage("left-ul");
                        }
                    }
                } else {
                    toPaint = t.getImage("left");
                }
            } else {
                Types.TERRAIN tl = board.getTerrainAt(i, j - 1);
                Types.TERRAIN td = board.getTerrainAt(i + 1, j);
                Types.TERRAIN tld = board.getTerrainAt(i + 1, j - 1);
                if (i < gridSize-1 && j > 0 && (tld == SHALLOW_WATER || tld == DEEP_WATER) &&
                        (tl != SHALLOW_WATER && tl != DEEP_WATER) &&
                        (td != SHALLOW_WATER && td != DEEP_WATER)) {
                    toPaint = t.getImage("dl");
                } else {
                    toPaint = t.getImage(null);
                }
            }
        }
        return toPaint;
    }

    // TODO: can draw more effects of actions, e.g. healing, disband
    void paintEffects(Graphics2D g) {
        if (effectDrawingIdx != -1) {
            Image effectImage = null;
            if (effectType == EFFECT.EXPLOSION) {
                if (effectDrawingIdx >= explosionEffect.length) {
                    effectDrawingIdx = -1;  // Finished
                    effectPositions.clear();
                    return;
                }
                effectImage = explosionEffect[effectDrawingIdx];
            } else if (effectType == EFFECT.PIERCE) {
                if (effectDrawingIdx >= pierceEffect.length) {
                    effectDrawingIdx = -1;  // Finished
                    effectPositions.clear();
                    return;
                }
                effectImage = pierceEffect[effectDrawingIdx];
            } else if (effectType == EFFECT.SLASH) {
                if (effectDrawingIdx >= slashEffect[effectTribeIdx].length) {
                    effectDrawingIdx = -1;  // Finished
                    effectPositions.clear();
                    return;
                }
                effectImage = slashEffect[effectTribeIdx][effectDrawingIdx];
            } else if (effectType == EFFECT.HEAL) {
                if (effectDrawingIdx >= healEffect[effectTribeIdx].length) {
                    effectDrawingIdx = -1;  // Finished
                    effectPositions.clear();
                    return;
                }
                effectImage = healEffect[effectTribeIdx][effectDrawingIdx];
            } else if (effectType == EFFECT.CONVERT) {
                if (effectDrawingIdx >= convertEffect[effectTribeIdx].length) {
                    effectDrawingIdx = -1;  // Finished
                    effectPositions.clear();
                    return;
                }
                effectImage = convertEffect[effectTribeIdx][effectDrawingIdx];
            }

            // Draw effect
            if (effectImage != null) {
                for (Vector2d effectPosition : effectPositions) {
                    Vector2d rotated = rotatePoint(1.0 * effectPosition.x / CELL_SIZE, 1.0 * effectPosition.y / CELL_SIZE);
                    paintImage(g, rotated.x + CELL_SIZE / 5, rotated.y - CELL_SIZE / 2, effectImage, CELL_SIZE, panTranslate);
                    if (effectType == EFFECT.SLASH || effectType == EFFECT.CONVERT) {
                        int x = rotated.x + CELL_SIZE / 5;
                        int y = rotated.y - CELL_SIZE / 2;
                        paintImageRotated(g, x, y, effectImage, CELL_SIZE, panTranslate, Math.PI / 2, x + CELL_SIZE/2, y + CELL_SIZE/2);
                    }
                }
                effectDrawingIdx++;
            } else {
                effectDrawingIdx = -1;
                effectPositions.clear();
            }
        }
    }

    void paintAction(UnitAction a) {
        if (a.getActionType() == ATTACK || a.getActionType() == CONVERT || a.getActionType() == HEAL_OTHERS) {  // These are the actions currently animated
            animationSpeed.clear();
            actionAnimationUnitsTribe.clear();
            sourceTargetAnimationInfo.clear();
            animatedAction = null;

            Unit source = (Unit) gameState.getBoard().getActor(a.getUnitId());
            Image weapon1 = source.getType().getWeaponImage(source.getTribeId());

            if (weapon1 == null) {
                animatedAction = a;
                return;
            }

            if (weapon1 != null) {
                // Pause the game, paint this weapon image travelling from attacker to target
                game.setAnimationPaused(true);
                Pair<Image, Vector2d> sourceAnimationInfo = new Pair<>(weapon1, new Vector2d(source.getPosition().y * CELL_SIZE, source.getPosition().x * CELL_SIZE));
                ArrayList<Unit> targets = new ArrayList<>();
                Image weapon2 = null;

                if (a.getActionType() == ATTACK) {
                    Unit t = (Unit) gameState.getBoard().getActor(((Attack) a).getTargetId());
                    targets.add(t);
                    weapon2 = t.getType().getWeaponImage(t.getTribeId());  // units can retaliate in Attack actions

                    this.effectType = EFFECT.SLASH;
                    if (source.getType() == CATAPULT || source.getType() == SCOUT || source.getType() == BOMBER) {
                        this.effectType = EFFECT.EXPLOSION;
                    } else if (source.getType() == ARCHER || source.getType() == RAFT) {
                        this.effectType = EFFECT.PIERCE;
                    }
                } else if (a.getActionType() == CONVERT) {
                    this.effectType = EFFECT.CONVERT;
                    Unit target = (Unit) gameState.getBoard().getActor(((Convert) a).getTargetId());
                    targets.add(target);
                } else {
                    // Heal Others
                    this.effectType = EFFECT.HEAL;
                    ArrayList<Unit> ts = ((HealOthers) a).getTargets(gameState);
                    targets.addAll(ts);
                }

                for (Unit target: targets) {
                    Pair<Image, Vector2d> targetAnimationInfo = new Pair<>(weapon2, new Vector2d(target.getPosition().y * CELL_SIZE, target.getPosition().x * CELL_SIZE));
                    double distance = manhattanDistance(source.getPosition(), target.getPosition());
                    animationSpeed.add(Math.max(2.0, Math.min(CELL_SIZE / 6.0, CELL_SIZE / (10.0 + distance * 3.0))));
                    actionAnimationUnitsTribe.add(new Pair<>(source.getTribeId(), target.getTribeId()));
                    sourceTargetAnimationInfo.add(new Pair<>(
                            new Pair<>(sourceAnimationInfo.getFirst(), sourceAnimationInfo.getSecond().copy()),
                            targetAnimationInfo));
                }
                if (targets.size() > 0) {
                    this.animatedAction = a;
                }
            }
        }
    }

    private void paintActionAnimations(Graphics2D g) {
        if (sourceTargetAnimationInfo.size() > 0) {
            ArrayList<Integer> finished = new ArrayList<>();
            for (int i = 0; i < sourceTargetAnimationInfo.size(); i++) {
                Pair<Image,Vector2d> source = sourceTargetAnimationInfo.get(i).getFirst();
                Pair<Image,Vector2d> target = sourceTargetAnimationInfo.get(i).getSecond();

                // Sprite not yet reached its destination, paint current and calculate next
                Vector2d currentPosition = source.getSecond().copy();

                double dxToTarget = target.getSecond().x - currentPosition.x;
                double dyToTarget = target.getSecond().y - currentPosition.y;
                double distanceToTarget = Math.hypot(dxToTarget, dyToTarget);
                double step = animationSpeed.get(i);
                if (distanceToTarget <= step) {
                    source.getSecond().set(target.getSecond());
                } else {
                    int xDir = (int)Math.round(dxToTarget / distanceToTarget * step);
                    int yDir = (int)Math.round(dyToTarget / distanceToTarget * step);
                    if (xDir == 0 && dxToTarget != 0) xDir = (int)Math.signum(dxToTarget);
                    if (yDir == 0 && dyToTarget != 0) yDir = (int)Math.signum(dyToTarget);
                    source.getSecond().add(xDir, yDir);
                }
                Vector2d nextPosition = source.getSecond().copy();

                // Rotate image in direction of travel
                double dx = nextPosition.x - currentPosition.x;
                double dy = nextPosition.y - currentPosition.y;
                double imageAngleRad = Math.atan2(dx, dy);// + Math.toRadians(180);

                Vector2d rotated = rotatePoint(1.0 * currentPosition.x / CELL_SIZE, 1.0 * currentPosition.y / CELL_SIZE);
                int x = rotated.x + CELL_SIZE / 2;
                int y = rotated.y - CELL_SIZE / 4;
                paintImageRotated(g, x, y, source.getFirst(), CELL_SIZE / 2, panTranslate, imageAngleRad, x + CELL_SIZE/4, y + CELL_SIZE/4);

                if (nextPosition.equals(target.getSecond())) {
                    // Reached destination, no more drawing. Reset animation variables and unpause game, unless retaliation happening

                    // Draw end of animation effect
                    effectPositions.add(target.getSecond());
                    effectTribeIdx = actionAnimationUnitsTribe.get(i).getFirst();
                    effectDrawingIdx = 0;

                    boolean finishAnimation = true;
                    if (animatedAction.getActionType() == ATTACK && target.getFirst() != null) {
                        boolean retaliates = new AttackCommand().isRetaliation((Attack) animatedAction, gameState);
                        if(retaliates) {
                            // Retaliating! Reset variables to target's attack
                            Vector2d startPosition = target.getSecond().copy();
                            Vector2d targetPosition = board.getActor(animatedAction.getUnitId()).getPosition().copy();
                            Vector2d endPosition = new Vector2d(targetPosition.y * CELL_SIZE, targetPosition.x * CELL_SIZE);
                            source = new Pair<>(target.getFirst(), startPosition);
                            target = new Pair<>(null, endPosition);
                            actionAnimationUnitsTribe.get(i).swap();
                            sourceTargetAnimationInfo.set(i, new Pair<>(source, target));
                            finishAnimation = false;
                        }
                    }

                    if(finishAnimation)
                    {
                        finished.add(i);
                    }
                }
            }
            finished.sort(Collections.reverseOrder());
            for (int i: finished) {
                sourceTargetAnimationInfo.remove(i);
                animationSpeed.remove(i);
                actionAnimationUnitsTribe.remove(i);
            }
        } else {
            if (effectDrawingIdx == -1 || !GUI_DRAW_EFFECTS) {
                game.setAnimationPaused(false);
            }
        }
    }

    Action getAnimatedAction() {
        if (sourceTargetAnimationInfo.size() == 0 && animatedAction != null) {
            Action a = animatedAction.copy();
            animatedAction = null;
            return a;
        }
        return null;
    }
}

