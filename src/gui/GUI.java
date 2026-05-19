package gui;

import core.Constants;
import core.Types;
import core.actions.tribeactions.EndTurn;
import core.actions.tribeactions.TribeAction;
import core.actions.unitactions.*;
import core.actors.units.Unit;
import core.game.Game;
import core.game.GameState;
import core.actions.Action;
import players.ActionController;
import players.HumanAgent;
import utils.Vector2d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static core.Constants.*;
import static core.Types.ACTION.*;

public class GUI extends JFrame {
    private static final int MIN_CELL_SIZE = 8;
    private static final int MAX_CELL_SIZE = 160;
    private static final Color PANEL_BG = new Color(244, 239, 226);
    private static final Color SECTION_BG = new Color(252, 248, 236);
    private static final Color SECTION_BORDER = new Color(171, 148, 105);
    private static final Color TEXT_PRIMARY = new Color(54, 47, 36);
    private static final Color TEXT_MUTED = new Color(95, 83, 62);
    private static final Color BUTTON_BG = new Color(78, 112, 91);
    private static final Color BUTTON_FG = new Color(250, 247, 235);

    private JLabel gameSummaryInfo;
    private JLabel runInfo;
    private int levelingUp = 0;

    private Game game;
    private GameState gs;

    private boolean pauseAfterTurn = false;  // If game should automatically pause after one turn (of one tribe) is played
    private boolean pauseAfterTick = false;  // If game should automatically pause after one tick (all tribes) is played
    private boolean pauseAfterAction = false;  // If game should pause after the next executed action

    private WindowInput wi;
    private ActionController ac;
    private Examine lastExamineAction;

    private GameView boardView;
    private TribeView tribeView;
    private TechView techView;
    private GameLogView gameLogView;
    private InfoView infoView;

    // Zoomed screen dragging vars
    private Vector2d lastDrag, panTranslate;
    private boolean draggedSincePress = false;
    private int pendingWheelRotation = 0;
    private Point2D pendingWheelMouseLocation = null;
    private final Timer zoomWheelTimer;

    public static double screenDiagonal;
    double scale = 1;

    /**
     * Constructor
     * @param title Title of the window.
     */
    public GUI(Game game, String title, WindowInput wi, ActionController ac, boolean closeAppOnClosingWindow) {
        super(title);

        try {
            UIManager.setLookAndFeel( UIManager.getCrossPlatformLookAndFeelClassName() );
        } catch (Exception e) {
            e.printStackTrace();
        }

        Rectangle rect = GraphicsEnvironment.getLocalGraphicsEnvironment().getMaximumWindowBounds();
        screenDiagonal = Math.sqrt(rect.width*rect.width + rect.height* rect.height);

        CELL_SIZE = (int)(0.038*screenDiagonal*scale);
        CELL_SIZE = Math.max(MIN_CELL_SIZE, Math.min(MAX_CELL_SIZE, CELL_SIZE));
        GUI_GAME_VIEW_SIZE = (int)(0.36*screenDiagonal*scale);
        GUI_MIN_PAN = (int)(0.015*screenDiagonal*scale);
        GUI_COMP_SPACING = (int)(0.0045*screenDiagonal*scale);
        GUI_CITY_TAG_WIDTH = (int)(0.009*screenDiagonal*scale);
        GUI_SIDE_PANEL_WIDTH = (int)(0.25*screenDiagonal*scale);
        GUI_INFO_PANEL_HEIGHT = (int)(0.18*screenDiagonal*scale);
        GUI_ACTION_PANEL_HEIGHT = (int)(0.045*screenDiagonal*scale);
        GUI_TECH_PANEL_HEIGHT = (int)(0.16*screenDiagonal*scale);
        GUI_TECH_PANEL_FULL_SIZE = (int)(GUI_TECH_PANEL_FULL_SIZE*(1/scale));

        this.ac = ac;
        this.wi = wi;
        this.game = game;
        this.zoomWheelTimer = new Timer(16, e -> applyPendingZoom());
        this.zoomWheelTimer.setRepeats(false);

        infoView = new InfoView(ac);
        panTranslate = new Vector2d(0,0);
        boardView = new GameView(game, infoView, panTranslate);

        setLayout(new BorderLayout(GUI_COMP_SPACING, 0));

        // Main panel definition
        JPanel mainPanel = createGamePanel();
        JPanel sidePanel = createSidePanel();

        getContentPane().add(mainPanel, BorderLayout.CENTER);
        getContentPane().add(sidePanel, BorderLayout.EAST);

        // Frame properties
        pack();
        this.setVisible(true);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        if(closeAppOnClosingWindow){
            setDefaultCloseOperation(EXIT_ON_CLOSE);
        }
        repaint();
    }


    private JPanel createGamePanel()
    {
        JPanel mainPanel = new JPanel();

        mainPanel.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (draggedSincePress) {
                    draggedSincePress = false;
                    return;
                }

                //Only provide information if clicking on a visible tile
                Vector2d translate = boardView.getPanTranslate();
                Vector2d ep = new Vector2d(e.getX() - translate.x, e.getY() - translate.y);
                Vector2d p = GameView.rotatePointReverse((int)ep.x, (int)ep.y);

                boolean canControlActiveTribe = game.getPlayers()[gs.getActiveTribeID()] instanceof HumanAgent ||
                        !DISABLE_NON_HUMAN_GRID_HIGHLIGHT;

                // Tile inspection should always work. Action execution remains restricted to controllable turns.
                if (canControlActiveTribe) {
                    Action candidate = getActionAt(p.x, p.y, infoView.getHighlightX(), infoView.getHighlightY());
                    if (candidate != null) {
                        int n = 0;
                        if (candidate.getActionType() == DISBAND) {  // These actions needs confirmation before executing
                            n = JOptionPane.showConfirmDialog(mainPanel,
                                    "Confirm action " + candidate.toString(),
                                    "Are you sure?",
                                    JOptionPane.YES_NO_OPTION,
                                    JOptionPane.QUESTION_MESSAGE);
                        }
                        if (candidate instanceof TribeAction) {
                            ((TribeAction) candidate).setTribeId(gs.getActiveTribeID());
                        }
                        if (n == 0) {
                            ac.addAction(candidate, gs);
                        }
                        infoView.resetHighlight();
                        return;
                    }
                }

                // Otherwise highlight new cell
                infoView.setHighlight(p.x, p.y);
//                    System.out.println("Highlighting: " + (int)p.getX() + " " + (int)p.getY());
            }

            @Override
            public void mousePressed(MouseEvent e) {
                lastDrag = new Vector2d(e.getX(), e.getY());
                draggedSincePress = false;
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                lastDrag = null;
            }

            @Override
            public void mouseEntered(MouseEvent e) {

            }

            @Override
            public void mouseExited(MouseEvent e) {

            }
        });
        mainPanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (lastDrag == null) {
                    lastDrag = new Vector2d(e.getX(), e.getY());
                    return;
                }

                Vector2d currentDrag = new Vector2d(e.getX(), e.getY());
                Vector2d delta = new Vector2d(currentDrag.x - lastDrag.x, currentDrag.y - lastDrag.y);
                if (delta.dist(0, 0) >= 1) {
                    draggedSincePress = true;
                    boardView.updatePan(delta);
                    infoView.resetHighlight();
                    boardView.repaint();
                }
                lastDrag = currentDrag;
            }
        });
        mainPanel.addMouseWheelListener(e -> {
            pendingWheelRotation += e.getWheelRotation();
            pendingWheelMouseLocation = e.getPoint();
            if (zoomWheelTimer.isRunning()) {
                zoomWheelTimer.restart();
            } else {
                zoomWheelTimer.start();
            }
        });

        mainPanel.setLayout(new BorderLayout());
        mainPanel.setPreferredSize(new Dimension(GUI_GAME_VIEW_SIZE, GUI_GAME_VIEW_SIZE));
        mainPanel.add(boardView, BorderLayout.CENTER);

        return mainPanel;
    }

    private void applyPendingZoom() {
        if (pendingWheelRotation == 0 || pendingWheelMouseLocation == null) {
            return;
        }

        Point2D mouseLocation = pendingWheelMouseLocation;
        pendingWheelMouseLocation = null;

        Vector2d currentPan = boardView.getPanTranslate();
        Point2D.Double gridPoint = GameView.rotatePointReverseExact(
                mouseLocation.getX() - currentPan.x,
                mouseLocation.getY() - currentPan.y);

        int oldCellSize = CELL_SIZE;
        int requestedCellSize = CELL_SIZE - pendingWheelRotation * GUI_ZOOM_FACTOR;
        pendingWheelRotation = 0;
        CELL_SIZE = Math.max(MIN_CELL_SIZE, Math.min(MAX_CELL_SIZE, requestedCellSize));

        if (CELL_SIZE == oldCellSize) {
            return;
        }

        Point2D.Double newScreenPoint = GameView.rotatePointExact(gridPoint.x, gridPoint.y);
        Vector2d desiredPan = new Vector2d(
                (int) Math.round(mouseLocation.getX() - newScreenPoint.x),
                (int) Math.round(mouseLocation.getY() - newScreenPoint.y));
        boardView.updatePan(new Vector2d(desiredPan.x - currentPan.x, desiredPan.y - currentPan.y));
        boardView.repaint();
    }

    /**
     * Creates side panel supplement for game view, containing:
     * - turn info
     * - active tribe info
     * - other info (Examine action / tribe win status),
     * - info on game grid highlights (units, terrain etc.) or research in tech tree highlights
     * - actions available for grid highlight
     * - all tribes points ranking
     * - tech tree view
     * - buttons to "End Turn", "Play Turn" (game paused after tribe's turn), "Play Tick" (game paused after all tribe's
     *      turns in current tick), "Pause/Resume" (to pause/resume game at any point), all keeping GUI responsive
     * - quick select buttons (all units / units of type / cities of a tribe) for info display TODO
     * @return JPanel containing all the sub components
     */
    private JPanel createSidePanel()
    {
        JPanel sidePanel = new JPanel(new GridBagLayout());
        sidePanel.setBackground(PANEL_BG);
        sidePanel.setPreferredSize(new Dimension(GUI_SIDE_PANEL_WIDTH, GUI_GAME_VIEW_SIZE));
        GridBagConstraints c = fullWidthConstraints();
        c.insets = new Insets(3, 0, 3, 0);
        int row = 0;

        JLabel appTitle = new JLabel("TribesTopia");
        Font textFont = new Font(Font.SERIF, Font.PLAIN, 15);
        appTitle.setFont(new Font(Font.SERIF, Font.BOLD, 20));
        appTitle.setForeground(TEXT_PRIMARY);
        appTitle.setHorizontalAlignment(SwingConstants.CENTER);

        runInfo = createInfoLabel(game.getRunDescription());
        runInfo.setFont(new Font(Font.SERIF, Font.PLAIN, 12));
        runInfo.setForeground(TEXT_MUTED);
        gameSummaryInfo = createInfoLabel("Turn 0\nActing: -\nResult: In progress");
        gameSummaryInfo.setFont(textFont);

        JTabbedPane tribeResearchInfo = new JTabbedPane();
        tribeView = new TribeView(game);
        techView = new TechView(game, ac, infoView);
        gameLogView = new GameLogView();
        tribeResearchInfo.setPreferredSize(new Dimension(GUI_SIDE_PANEL_WIDTH, GUI_TECH_PANEL_HEIGHT));
        tribeResearchInfo.add("Tribe Info", tribeView);
        tribeResearchInfo.add("Game Log", gameLogView);
        tribeResearchInfo.add("Tech Tree", techView);

        c.gridy = row++;
        sidePanel.add(appTitle, c);

        JPanel gameStatus = createCompactSection("Game");
        addCompactRow(gameStatus, gameSummaryInfo, 0);
        addCompactRow(gameStatus, runInfo, 1);
        c.gridy = row++;
        sidePanel.add(gameStatus, c);

        c.gridy = row++;
        c.weighty = 0.3;
        c.fill = GridBagConstraints.BOTH;
        sidePanel.add(tribeResearchInfo, c);
        c.weighty = 0;
        c.fill = GridBagConstraints.HORIZONTAL;

        JPanel controls = createSection("Playback");
        JPanel buttons = new JPanel(new GridLayout(0, 3, 4, 4));
        buttons.setOpaque(false);
        JButton endTurn = new JButton("End Turn");
        endTurn.addActionListener(e -> ac.addAction(new EndTurn(gs.getActiveTribeID()), gs));
        JButton playTurn = new JButton("Turn");
        playTurn.addActionListener(e -> { pauseAfterTurn = true; game.setPaused(false); });
        JButton playTick = new JButton("Round");
        playTick.addActionListener(e -> { pauseAfterTick = true; game.setPaused(false); });
        JButton nextAction = new JButton("Action");
        nextAction.addActionListener(e -> { pauseAfterAction = true; game.setPaused(false); });
        JButton pause = new JButton("Pause");
        pause.addActionListener(e -> {
            if (game.isPaused()) {
                game.setPaused(false);
                pause.setText("Pause");
            } else {
                game.setPaused(true);
                pause.setText("Resume");
            }
        });

        for (JButton button : new JButton[]{endTurn, nextAction, playTurn, playTick, pause}) {
            styleButton(button);
            buttons.add(button);
        }
        addSectionRow(controls, buttons, 0, 0.0);
        addWatchControls(controls, 1);
        c.gridy = row++;
        sidePanel.add(controls, c);

        infoView.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(SECTION_BORDER),
                "Selected Tile", 0, 0,
                new Font(Font.SERIF, Font.BOLD, 12), TEXT_PRIMARY));
        infoView.setBackground(SECTION_BG);
        c.gridy = row++;
        sidePanel.add(infoView, c);

        c.gridy = row;
        c.weighty = 1.0;
        c.fill = GridBagConstraints.BOTH;
        sidePanel.add(Box.createVerticalGlue(), c);

        return sidePanel;
    }

    /**
     * Panel containing functionality for interacting with the framework and getting high-level information, including:
     * - Game setup:
     *      - Choosing which players should play in the next game TODO
     *      - Choosing which tribe should be associated with each player in the next game TODO
     *      - Choosing and editing map for next game (text view) TODO
     *      - Choosing game seed for next game TODO
     *      - Start/Restart/End game buttons TODO
     * - Debugging:
     *      - observability toggle TODO
     *      - visuals on/off TODO
     *      - results printout for all games in the same run (if multiple) TODO
     *      - save game toggle TODO
     *      - add comments to saved game files TODO
     *      - action history display TODO
     *      - change game configuration? TODO
     * @return
     */
    private void addWatchControls(JPanel controls, int startRow) {
        JLabel speedLabel = new JLabel("Delay: " + FRAME_DELAY + " ms");
        speedLabel.setFont(new Font(Font.SERIF, Font.PLAIN, 12));
        speedLabel.setForeground(TEXT_MUTED);
        JSlider speedSlider = new JSlider(0, 1000, FRAME_DELAY);
        speedSlider.setMajorTickSpacing(250);
        speedSlider.setMinorTickSpacing(50);
        speedSlider.setPaintTicks(true);
        speedSlider.setOpaque(false);
        speedSlider.addChangeListener(e -> {
            Constants.FRAME_DELAY = speedSlider.getValue();
            speedLabel.setText("Delay: " + Constants.FRAME_DELAY + " ms");
        });

        JCheckBox followActive = new JCheckBox("Follow active tribe", Constants.GUI_PAN_TO_TRIBE);
        followActive.setOpaque(false);
        followActive.setFont(new Font(Font.SERIF, Font.PLAIN, 12));
        followActive.setForeground(TEXT_MUTED);
        followActive.addActionListener(e -> Constants.GUI_PAN_TO_TRIBE = followActive.isSelected());

        JButton saveSnapshot = new JButton("Save");
        styleButton(saveSnapshot);
        saveSnapshot.addActionListener(e -> {
            try {
                game.writeSnapshot();
                gameLogView.addEntry("Snapshot saved.");
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, ex.getMessage(), "Save Snapshot Failed", JOptionPane.ERROR_MESSAGE);
            }
        });

        JPanel watchRow = new JPanel(new BorderLayout(4, 0));
        watchRow.setOpaque(false);
        watchRow.add(followActive, BorderLayout.CENTER);
        watchRow.add(saveSnapshot, BorderLayout.EAST);

        addSectionRow(controls, speedLabel, startRow, 0.0);
        addSectionRow(controls, speedSlider, startRow + 1, 0.0);
        addSectionRow(controls, watchRow, startRow + 2, 0.0);
    }

    private GridBagConstraints fullWidthConstraints() {
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.weightx = 1.0;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.anchor = GridBagConstraints.NORTH;
        return c;
    }

    private JPanel createSection(String title) {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBackground(SECTION_BG);
        panel.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(SECTION_BORDER),
                title, 0, 0,
                new Font(Font.SERIF, Font.BOLD, 12), TEXT_PRIMARY));
        return panel;
    }

    private JPanel createCompactSection(String title) {
        JPanel panel = createSection(title);
        panel.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(SECTION_BORDER),
                title, 0, 0,
                new Font(Font.SERIF, Font.BOLD, 12), TEXT_PRIMARY));
        return panel;
    }

    private void addSectionRow(JPanel panel, Component component, int row, double weighty) {
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = row;
        c.weightx = 1.0;
        c.weighty = weighty;
        c.fill = weighty > 0 ? GridBagConstraints.BOTH : GridBagConstraints.HORIZONTAL;
        c.anchor = GridBagConstraints.NORTHWEST;
        c.insets = new Insets(2, 5, 2, 5);
        panel.add(component, c);
    }

    private void addCompactRow(JPanel panel, Component component, int row) {
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = row;
        c.weightx = 1.0;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.anchor = GridBagConstraints.NORTHWEST;
        c.insets = new Insets(row == 0 ? 0 : 1, 5, 0, 5);
        panel.add(component, c);
    }

    private void styleButton(JButton button) {
        button.setBackground(BUTTON_BG);
        button.setForeground(BUTTON_FG);
        button.setFocusPainted(false);
        button.setFont(new Font(Font.SERIF, Font.BOLD, 12));
        button.setMargin(new Insets(2, 6, 2, 6));
        button.setOpaque(true);
    }

    private JLabel createInfoLabel(String text) {
        JLabel label = new JLabel();
        label.setVerticalAlignment(SwingConstants.TOP);
        label.setForeground(TEXT_PRIMARY);
        setWrappedInfo(label, text);
        return label;
    }

    private void setWrappedInfo(JLabel label, String text) {
        int width = Math.max(120, GUI_SIDE_PANEL_WIDTH - 50);
        label.setText("<html><body style='width:" + width + "px'>" + escapeHtml(text).replace("\n", "<br/>") + "</body></html>");
        label.setToolTipText(text);
        label.revalidate();
    }

    private String escapeHtml(String text) {
        return text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;");
    }


    /**
     * Paints the GUI, to be called at every game tick.
     */
    public void update(GameState gs, Action a) {
//        if (this.gs == null || this.gs.getTick() != gs.getTick()) {
//            // Tick change
//            HashMap<Integer, ArrayList<Action>> tick = new HashMap<>();
//            for (int i = 0; i < game.getPlayers().length; i++) {
//                tick.put(i, new ArrayList<>());
//            }
//            actionHistory.add(tick);
//        }
        if (this.gs == null || this.gs.getActiveTribeID() != gs.getActiveTribeID()) {
            // Tribe change
            infoView.refreshHighlight();  // Refresh inspector/actions for the new active tribe.
            if(Constants.GUI_PAN_TO_TRIBE)
                boardView.setPanToTribe(gs);  // Pan camera to tribe capital
            ac.reset();  // Clear action queue
        }

        if (a != null)
        {
            gameLogView.addAction(gs.getTick(), gs.getActiveTribeID(), a);
            // Display result of Examine action
            if (a.getActionType() == EXAMINE) {
                lastExamineAction = (Examine) a;
            }
            // Draw animations for these actions
            if (a.getActionType() == ATTACK ||
                    a.getActionType() == CONVERT ||
                    a.getActionType() == HEAL_OTHERS) {
                boardView.paintAction((UnitAction)a);
            }
        }

//        if (this.gs != null) {
//            if (a != null) {
//                actionHistory.get(this.gs.getTick()).get(this.gs.getActiveTribeID()).add(a);
//            }
//            stateHistory.add(this.gs);
//        }
        this.gs = gs;
        if (gs.isLevelingUp()) this.levelingUp++;
        else this.levelingUp = 0;
        performUpdate();

        // Check if city is levelling up, pop up dialogue to choose options if human agent
        if (this.levelingUp == 1 && game.getPlayers()[gs.getActiveTribeID()] instanceof HumanAgent) {
            int n = -1;
            java.util.List<Action> levelUpActions = new java.util.ArrayList<>();
            HashMap<Integer, ArrayList<Action>> actions = gs.getCityActions();
            for (Map.Entry<Integer, ArrayList<Action>> e : actions.entrySet()) {
                for (Action action : e.getValue()) {
                    if (action.getActionType() == LEVEL_UP) {
                        levelUpActions.add(action);
                    }
                }
            }
            if (levelUpActions.isEmpty()) {
                return;
            }
            Object[] options = levelUpActions.stream().map(Action::toString).toArray(String[]::new);
            while (n == -1) {
                n = JOptionPane.showOptionDialog(this, //parent container of JOptionPane
                        "City is levelling up!",
                        "Level Up",
                        JOptionPane.YES_NO_OPTION,
                        JOptionPane.QUESTION_MESSAGE,
                        null,  //do not use a custom Icon
                        options,  //the titles of buttons
                        options[0]);  //default button title
            }
            ac.addAction(levelUpActions.get(n), gs);
        }
    }

    /**
     * Retrieves action at specific location given by (actionX, actionY) coordinates, to be performed by
     * unit at coordinates (unitX, unitY).
     */
    private Action getActionAt(int actionX, int actionY, int unitX, int unitY) {
        if (!infoView.clickedTwice()) { // Only return action if we're highlighting the unit and not underneath it
            HashMap<Integer, ArrayList<Action>> possibleActions = gs.getUnitActions();
            for (Map.Entry<Integer, ArrayList<Action>> e : possibleActions.entrySet()) {
                Unit u = (Unit) gs.getActor(e.getKey());
                for (Action a : e.getValue()) {
                    Vector2d pos = getActionPosition(gs, a);
                    if (pos != null && pos.x == actionY && pos.y == actionX) {
                        if ((a.getActionType() == CAPTURE || a.getActionType() == EXAMINE) ||  // These actions don't need the unit highlighted
                                u.getPosition().x == unitY && u.getPosition().y == unitX) {
                            return a;
                        }
                    }
                }
            }
        }
        return null;
    }

    private void performUpdate() {
        boardView.refresh(gs);
        tribeView.refresh(gs);
        techView.refresh(game.copyGameState());
        infoView.refresh(gs);
        if (gs.getActiveTribe() != null) {
            Types.RESULT winStatus = gs.getTribeWinStatus(gs.getActiveTribeID());
            String resultText = winStatus == Types.RESULT.INCOMPLETE ? "In progress" : winStatus.toString();
            setWrappedInfo(gameSummaryInfo,
                    "Turn " + gs.getTick() + (game.isPaused() ? " | Paused" : "")
                            + "\nActing: " + gs.getActiveTribe().getName()
                            + "\nResult: " + resultText);
            if (winStatus != Types.RESULT.INCOMPLETE) {
                // Already shown in the Game summary.
            } else if (lastExamineAction != null && lastExamineAction.getBonus() != null) {
                gameLogView.addEntry("Ruins: " + lastExamineAction.getBonus());
                lastExamineAction = null;
            }
        }
        repaint();
    }

    public static Vector2d getActionPosition(GameState gs, Action a) {
        Vector2d pos = null;
        if (a.getActionType() == MOVE) {
            pos = new Vector2d(((Move) a).getDestination().x, ((Move) a).getDestination().y);
        } else if (a.getActionType() == ATTACK) {
            Unit target = (Unit) gs.getActor(((Attack) a).getTargetId());
            pos = target.getPosition();
        } else if (a.getActionType() == RECOVER) {
            Unit u = (Unit) gs.getActor(((UnitAction) a).getUnitId());
            pos = u.getPosition();
        } else if (a.getActionType() == CAPTURE || a.getActionType() == EXAMINE) {
            Unit u = (Unit) gs.getActor(((UnitAction) a).getUnitId());
            pos = new Vector2d(u.getPosition().x-1, u.getPosition().y);
        } else if (a.getActionType() == CONVERT) {
            Unit target = (Unit) gs.getActor(((Convert) a).getTargetId());
            pos = target.getPosition();
        }
        return pos;
    }

    public boolean isClosed() {
        return wi.windowClosed;
    }

    public boolean pauseAfterTurn() {
        return pauseAfterTurn;
    }

    public void setPauseAfterTurn(boolean p) {
        pauseAfterTurn = p;
    }

    public boolean pauseAfterTick() {
        return pauseAfterTick;
    }

    public void setPauseAfterTick(boolean p) {
        pauseAfterTick = p;
    }

    public boolean pauseAfterAction() {
        return pauseAfterAction;
    }

    public void setPauseAfterAction(boolean p) {
        pauseAfterAction = p;
    }

    public Action getAnimatedAction() {
        return boardView.getAnimatedAction();
    }

}
