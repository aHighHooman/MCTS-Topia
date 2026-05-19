package gui;

import core.TechnologyTree;
import core.Types;
import core.actions.cityactions.*;
import core.actions.tribeactions.BuildRoad;
import core.actions.tribeactions.ResearchTech;
import core.actions.unitactions.Disband;
import core.actions.unitactions.HealOthers;
import core.actions.unitactions.Upgrade;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.CarrierUnit;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import core.actions.Action;
import players.ActionController;
import utils.Vector2d;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import static core.Constants.*;
import static core.TribesConfig.VETERAN_KILLS;
import static gui.GameView.gridSize;
import static core.Types.ACTION.*;

@SuppressWarnings({"StringConcatenationInsideStringBufferAppend", "SuspiciousNameCombination"})
public class InfoView extends JComponent {

    // Dimensions of the window.
    private Dimension size;
    private JLabel textArea;

    private JButton actionBF, actionCF, actionD, actionGF, actionRG;
    private JButton[] actionB, actionS;
    private JButton actionRoad;
    private JButton actionResearch;
    private JButton actionHealOthers, actionDisband;
    private JButton actionUpgradeRammer, actionUpgradeScout, actionUpgradeBomber;
    private TribesActionListener listenerBF, listenerCF, listenerD, listenerGF, listenerRG;
    private TribesActionListener listenerS, listenerB;
    private TribesActionListener listenerResearch;
    private TribesActionListener listenerRoad;
    private TribesActionListener listenerHealOthers, listenerDisband;
    private TribesActionListener listenerUpgradeRammer, listenerUpgradeScout, listenerUpgradeBomber;
    private ActionController ac;

    private int highlightX, highlightY;
    private int highlightXprev, highlightYprev;
    private boolean updateHighlight, updateTechHighlight;
    Types.TECHNOLOGY techHighlight;

    private GameState gs;

    InfoView(ActionController ac)
    {
        this.size = new Dimension(GUI_SIDE_PANEL_WIDTH, GUI_INFO_PANEL_HEIGHT);
        this.ac = ac;
        highlightX = -1;
        highlightY = -1;
        highlightXprev = -1;
        highlightYprev = -1;

        textArea = new JLabel();
        Font textFont = new Font(Font.SERIF, Font.PLAIN, 12);
        textArea.setFont(textFont);
        textArea.setOpaque(false);
        textArea.setBorder(BorderFactory.createEmptyBorder(2, 6, 2, 6));
        textArea.setVerticalAlignment(SwingConstants.TOP);
        textArea.setText(formatInfoHtml("<font color='#5f533e'>Click a tile to inspect it.</font>"));

        JPanel actionPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 2, 1));
        actionPanel.setOpaque(false);

        // Simple actions: BurnForest, ClearForest, Destroy, GrowForest, GatherResource
        actionBF = new JButton("Burn");  // If forest
        listenerBF = new TribesActionListener("BurnForest");
        actionBF.addActionListener(listenerBF);
        actionBF.setVisible(false);
        actionCF = new JButton("Clear");  // If forest
        listenerCF = new TribesActionListener("ClearForest");
        actionCF.addActionListener(listenerCF);
        actionCF.setVisible(false);
        actionD = new JButton("Destroy");  // If building
        listenerD = new TribesActionListener("Destroy");
        actionD.addActionListener(listenerD);
        actionD.setVisible(false);
        actionGF = new JButton("Grow");  // If plain
        listenerGF = new TribesActionListener("GrowForest");
        actionGF.addActionListener(listenerGF);
        actionGF.setVisible(false);
        actionRG = new JButton("Gather");  // If resource
        listenerRG = new TribesActionListener("ResourceGathering");
        actionRG.addActionListener(listenerRG);
        actionRG.setVisible(false);
        actionPanel.add(actionRG);
        actionPanel.add(actionBF);
        actionPanel.add(actionCF);
        actionPanel.add(actionD);
        actionPanel.add(actionGF);

        // Complex actions: Build X, Spawn X
        int nBuildings = Types.BUILDING.values().length;
        actionB = new JButton[nBuildings];
        listenerB = new TribesActionListener("Build");
        for (int i = 0; i < nBuildings; i++) {
            actionB[i] = new JButton("Build " + Types.BUILDING.values()[i]);
            actionB[i].addActionListener(listenerB);
            actionB[i].setVisible(false);
            actionPanel.add(actionB[i]);
        }
        ArrayList<Types.UNIT> spawnableUnits = Types.UNIT.getSpawnableTypes();
        int nUnits = spawnableUnits.size();
        actionS = new JButton[nUnits];
        listenerS = new TribesActionListener("Spawn");
        for (int i = 0; i < nUnits; i++) {
            actionS[i] = new JButton("Spawn " + spawnableUnits.get(i));
            actionS[i].addActionListener(listenerS);
            actionS[i].setVisible(false);
            actionPanel.add(actionS[i]);
        }

        // Research action
        actionResearch = new JButton("Research");
        actionResearch.setVisible(false);
        listenerResearch = new TribesActionListener("Research");
        actionResearch.addActionListener(listenerResearch);
        actionPanel.add(actionResearch);

        // Build road action
        actionRoad = new JButton("Build Road");
        actionRoad.setVisible(false);
        listenerRoad = new TribesActionListener("BuildRoad");
        actionRoad.addActionListener(listenerRoad);
        actionPanel.add(actionRoad);

        // Unit actions
        actionHealOthers = new JButton("Heal Others");
        actionHealOthers.setVisible(false);
        listenerHealOthers = new TribesActionListener("HealOthers");
        actionHealOthers.addActionListener(listenerHealOthers);
        actionPanel.add(actionHealOthers);
        actionDisband = new JButton("Disband");
        actionDisband.setVisible(false);
        listenerDisband = new TribesActionListener("Disband");
        actionDisband.addActionListener(listenerDisband);
        actionPanel.add(actionDisband);
        actionUpgradeRammer = new JButton("Upgrade Rammer");
        actionUpgradeRammer.setVisible(false);
        listenerUpgradeRammer = new TribesActionListener("UpgradeRammer");
        actionUpgradeRammer.addActionListener(listenerUpgradeRammer);
        actionPanel.add(actionUpgradeRammer);
        actionUpgradeScout = new JButton("Upgrade Scout");
        actionUpgradeScout.setVisible(false);
        listenerUpgradeScout = new TribesActionListener("UpgradeScout");
        actionUpgradeScout.addActionListener(listenerUpgradeScout);
        actionPanel.add(actionUpgradeScout);
        actionUpgradeBomber = new JButton("Upgrade Bomber");
        actionUpgradeBomber.setVisible(false);
        listenerUpgradeBomber = new TribesActionListener("UpgradeBomber");
        actionUpgradeBomber.addActionListener(listenerUpgradeBomber);
        actionPanel.add(actionUpgradeBomber);

        this.setLayout(new BorderLayout(0, 1));
        this.add(textArea, BorderLayout.CENTER);
        this.add(actionPanel, BorderLayout.SOUTH);
    }


    public void paintComponent(Graphics gx)
    {
        super.paintComponent(gx);
    }

    private void updateInfoContents()
    {
        if (gs == null) return;

        Board board = gs.getBoard();

        if (highlightInGridBounds()) {

            Types.TERRAIN t = board.getTerrainAt(highlightY, highlightX);
            Types.RESOURCE r = board.getResourceAt(highlightY, highlightX);
            Types.BUILDING b = board.getBuildingAt(highlightY, highlightX);
            Unit u = board.getUnitAt(highlightY, highlightX);

            // t < r < b < u

            String s;

            if (u != null && !clickedTwice()) {
                // Unit is always on top, show this, unless clicked twice
                s = getUnitInfo(u);
            } else {
                s = "<h2>";
                boolean hasCompleteTitle = false;
                if (t != null) {
                    if (t == Types.TERRAIN.CITY) { // It's a city, show just this
                        s = getCityInfo();
                        hasCompleteTitle = true;
                    } else {
                        // Show everything else
                        s += t.toString();
                        if (r != null) {
                            // Resource next
                            s += ", " + r.toString();
                        }
                        if (b != null) {
                            // Buildings
                            s += ", " + b.toString();
                        }
                    }
                } else {
                    // Show buildings and resources
                    if (r != null) {
                        // Resource next
                        s += r.toString();
                        if (b != null) {
                            // Buildings
                            s += ", " + b.toString();
                        }
                        if (r == Types.RESOURCE.LIGHTHOUSE) {
                            s += lighthouseSeenByInfo(board, highlightY, highlightX);
                        }
                    } else if (b != null) {
                        // Buildings
                        s += b.toString();
                    }
                }
                if (!hasCompleteTitle) {
                    s += "</h2>";
                }
            }

            if (!textArea.getText().equals(s) && updateHighlight) {
                textArea.setText(formatInfoHtml(s));
                techHighlight = null;
                updateHighlight = false;
                updateButtons();
            }
        }
        updateTechButton();
    }

    private String lighthouseSeenByInfo(Board board, int x, int y) {
        ArrayList<Integer> discoverers = board.getLighthouseDiscoverers(x, y);
        if (discoverers.isEmpty()) {
            return "<br/>Seen by: none";
        }

        StringBuilder sb = new StringBuilder("<br/>Seen by: ");
        for (int i = 0; i < discoverers.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(Types.TRIBE.values()[discoverers.get(i)]);
        }
        return sb.toString();
    }

    private String getUnitInfo(Unit u) {
//        String img = u.getType().getImageStr(u.getTribeId());

        Tribe t = gs.getTribe(u.getTribeId());
        int idx = t.getType().getKey();

        StringBuilder sb = new StringBuilder();
        sb.append("<b>" + Types.TRIBE.values()[idx] + " " + u.getType() + "</b><br/>");
//        sb.append("<table border=\"0\"><tr><td><img src=\"file:" + img + "\"/></p></td><td>");
        sb.append("<font color='#5f533e'>From city " + u.getCityId() + "</font><br/>");
        if (u.isVeteran()) {
            sb.append("<b>Veteran unit.</b><br/>");
        } else {
            int kills = Math.min(u.getKills(), VETERAN_KILLS);
            sb.append(kills + "/" + VETERAN_KILLS + " kills to veteran.<br/>");
        }
//        sb.append("</td></tr></table>");
        sb.append("<table cellpadding='1' cellspacing='0'>");
        sb.append(row("Health", u.getCurrentHP() + "/" + u.getMaxHP()));
        sb.append(row("Attack", formatStat(u.getAttackValue())));
        sb.append(row("Defence", formatStat(u.getDefenceValue())));
        sb.append(row("Movement", u.MOV));
        sb.append(row("Range", u.RANGE));
        sb.append(row("Status", u.getStatus()));
        sb.append("</table>");

        if(u instanceof CarrierUnit)
            sb.append("<b>Land unit:</b> " + ((CarrierUnit)u).getBaseLandUnit() + "<br/>");

        return sb.toString();
    }

    private String getCityInfo() {
        Board board = gs.getBoard();
        int cityID = board.getCityIdAt(highlightY, highlightX);
        City c = (City) board.getActor(cityID);

        StringBuilder sb = new StringBuilder();
        if(c != null) {
            sb.append("<b>" + Types.TRIBE.values()[c.getTribeId()] + " city " + cityID + "</b><br/>");
//            sb.append("<table border=\"0\"><tr><td><img width=\"" + CELL_SIZE + "\" src=\"file:" + Types.TERRAIN.CITY.getImageStr() + "\"/></p></td><td>");
            sb.append("<table cellpadding='1' cellspacing='0'>");
            sb.append(row("Capital", c.isCapital()));
            sb.append(row("Points", c.getPointsWorth()));
            sb.append(row("Production", c.getProduction()));
            sb.append(row("Population", c.getPopulation() + "/" + c.getPopulation_need()));
            sb.append("</table>");
//            sb.append("</td></tr></table>");
        }
        return sb.toString();
    }

    private void updateButtons() {
        Board board = gs.getBoard();
        int cityID = board.getCityIdAt(highlightY, highlightX);
        Types.RESOURCE r = board.getResourceAt(highlightY, highlightX);
        Vector2d position = new Vector2d(highlightY, highlightX);
        Unit u = board.getUnitAt(highlightY, highlightX);
        resetButtonVisibility();

        if (board.getTribe(board.getActiveTribeID()).getTechTree().isResearched(Types.TECHNOLOGY.ROADS)) {
            ArrayList<Action> acts = gs.getTribeActions();
            for (Action a: acts) {
                if (a.getActionType() == BUILD_ROAD && ((BuildRoad) a).getPosition().equals(position)) {
                    actionRoad.setVisible(true);
                    listenerRoad.update(board.getActiveTribeID(), position, ac, gs);
                }
            }
        }

        if (cityID != -1) {
            City c = (City) gs.getBoard().getActor(cityID);
            if (c != null) {
                ArrayList<Action> acts = gs.getCityActions(c);

                boolean foundRG = false;
                boolean foundBF = false;
                boolean foundCF = false;
                boolean foundGF = false;
                boolean foundD = false;
                boolean[] foundS = new boolean[actionS.length];
                boolean[] foundB = new boolean[actionB.length];
                if (acts != null && acts.size() > 0) {
                    for (Action a : acts) {
                        if (a != null && ((CityAction) a).getTargetPos() != null &&
                                ((CityAction) a).getTargetPos().equals(position)) {
                            if (a.getActionType() == RESOURCE_GATHERING) {
                                if (((ResourceGathering) a).getResource().equals(r)) {
                                    listenerRG.update(cityID, position, ac, gs);
                                    listenerRG.setResource(r);
                                    foundRG = true;
                                }
                            } else if (a.getActionType() == BURN_FOREST) {
                                listenerBF.update(cityID, position, ac, gs);
                                foundBF = true;
                            } else if (a.getActionType() == CLEAR_FOREST) {
                                listenerCF.update(cityID, position, ac, gs);
                                foundCF = true;
                            } else if (a.getActionType() == GROW_FOREST) {
                                listenerGF.update(cityID, position, ac, gs);
                                foundGF = true;
                            } else if (a.getActionType() == DESTROY) {
                                listenerD.update(cityID, position, ac, gs);
                                foundD = true;
                                break;
                            } else if (a.getActionType() == SPAWN) {
                                Types.UNIT unitType = ((Spawn) a).getUnitType();
                                int idx = Types.UNIT.getSpawnableTypes().indexOf(unitType);
                                listenerS.update(cityID, position, ac, gs);
                                foundS[idx] = true;
                            } else if (a.getActionType() == BUILD) {
                                Types.BUILDING buildingType = ((Build) a).getBuildingType();
                                int idx = buildingType.getKey();
                                listenerB.update(cityID, position, ac, gs);
                                foundB[idx] = true;
                            }
                        }
                    }
                }
                actionRG.setVisible(foundRG);
                actionBF.setVisible(foundBF);
                actionCF.setVisible(foundCF);
                actionGF.setVisible(foundGF);
                actionD.setVisible(foundD);
                for (int i = 0; i < actionS.length; i++) {
                    actionS[i].setVisible(foundS[i]);
                }
                for (int i = 0; i < actionB.length; i++) {
                    actionB[i].setVisible(foundB[i]);
                }
            }
        }

        if (u != null) {
            ArrayList<Action> unitActions = gs.getUnitActions(u);
            if (unitActions != null && unitActions.size() > 0) {
                boolean foundHO = false;
                boolean foundD = false;
                boolean foundUpgradeRammer = false;
                boolean foundUpgradeScout = false;
                boolean foundUpgradeBomber = false;
                for (Action a : unitActions) {
                    if (a.getActionType() == HEAL_OTHERS) {
                        foundHO = true;
                        listenerHealOthers.update(u.getActorId(), ac, gs);
                    } else if (a.getActionType() == DISBAND) {
                        foundD = true;
                        listenerDisband.update(u.getActorId(), ac, gs);
                    } else if (a.getActionType() == UPGRADE_RAMMER) {
                        foundUpgradeRammer = true;
                        listenerUpgradeRammer.update(u.getActorId(), ac, gs);
                    } else if (a.getActionType() == UPGRADE_SCOUT) {
                        foundUpgradeScout = true;
                        listenerUpgradeScout.update(u.getActorId(), ac, gs);
                    } else if (a.getActionType() == UPGRADE_BOMBER) {
                        foundUpgradeBomber = true;
                        listenerUpgradeBomber.update(u.getActorId(), ac, gs);
                    }
                }
                actionHealOthers.setVisible(foundHO);
                actionDisband.setVisible(foundD);
                actionUpgradeRammer.setVisible(foundUpgradeRammer);
                actionUpgradeScout.setVisible(foundUpgradeScout);
                actionUpgradeBomber.setVisible(foundUpgradeBomber);
            }
        }
    }

    private void updateTechButton() {
        if (techHighlight != null && updateTechHighlight) {
            resetButtonVisibility();
            updateHighlight = false;
            actionResearch.setVisible(true);
            listenerResearch.update(techHighlight, ac, gs);

            Tribe t = gs.getActiveTribe();
            if (t != null) {
                TechnologyTree tt = t.getTechTree();
                if (tt != null) {
                    boolean researched = tt.isResearched(techHighlight);
                    boolean techRequirement = tt.isResearchable(techHighlight);
                    int starCost = techHighlight.getCost(t.getNumCities(), tt);
                    boolean starRequirement = t.getStars() >= starCost;
                    boolean researchable = techRequirement && starRequirement;

                    String txt = "";
                    if (!(researchable || researched)) {
                        actionResearch.setEnabled(false);
                        actionResearch.setText("Unavailable");
                        if (!techRequirement) txt += "Requires " + techHighlight.getParentTech().toString();
                        if (!starRequirement) {
                            if (!techRequirement) txt += "<br/>";
                            txt += "Not enough stars, " + t.getStars() + " of " + starCost + " required";
                        }
                        actionResearch.setToolTipText("<html>" + txt + "</html>");
                    } else {
                        if (researched) {
                            actionResearch.setEnabled(false);
                            txt = "Researched";
                            actionResearch.setText("Researched");
                        } else {
                            actionResearch.setEnabled(true);
                            txt = "Researchable";
                            actionResearch.setText("Research");
                        }
                    }

                    String fullTxt = textArea.getText().replaceAll("</*html>|</*head>|</*body>|\n|\r", "") + "<hr>" + txt;
                    textArea.setText(formatInfoHtml(fullTxt));
                }
            }
            updateTechHighlight = false;
        }
    }

    private void resetButtonVisibility(){
        actionBF.setVisible(false);
        actionCF.setVisible(false);
        actionD.setVisible(false);
        actionGF.setVisible(false);
        actionRG.setVisible(false);
        actionRoad.setVisible(false);
        actionResearch.setVisible(false);
        actionUpgradeRammer.setVisible(false);
        actionUpgradeScout.setVisible(false);
        actionUpgradeBomber.setVisible(false);
        actionHealOthers.setVisible(false);
        actionDisband.setVisible(false);
        for (JButton jb: actionB) {
            jb.setVisible(false);
        }
        for (JButton jb: actionS) {
            jb.setVisible(false);
        }
    }

    void refresh(GameState gs)
    {
        this.gs = gs;
        updateInfoContents();
    }

    /**
     * Gets the dimensions of the window.
     * @return the dimensions of the window.
     */
    public Dimension getPreferredSize() {
        return size;
    }

    public void setHighlight(int x, int y)
    {
        highlightXprev = highlightX;
        highlightYprev = highlightY;
        highlightX = x;
        highlightY = y;
        updateHighlight = true;
        updateInfoContents();
        revalidate();
        repaint();
    }

    public void resetHighlight() {
        highlightX = -1;
        highlightY = -1;
        highlightXprev = -1;
        highlightYprev = -1;
        techHighlight = null;

        updateHighlight = false;
        updateTechHighlight = false;

        // Reset highlight info text
        textArea.setText(formatInfoHtml("<font color='#5f533e'>Click a tile to inspect it.</font>"));

        // Reset actions, none available yet for this player
        resetButtonVisibility();

        repaint();
    }

    public void refreshHighlight() {
        if (highlightInGridBounds() || techHighlight != null) {
            updateHighlight = true;
            updateTechHighlight = techHighlight != null;
            updateInfoContents();
            revalidate();
            repaint();
        }
    }

    public int getHighlightX() {return highlightX;}
    public int getHighlightY() {return highlightY;}
    @SuppressWarnings("BooleanMethodIsAlwaysInverted")
    public boolean clickedTwice() {
        return highlightX != -1 && highlightX == highlightXprev && highlightY == highlightYprev;
    }
    public boolean highlightInGridBounds() {
        return highlightX > -1 && highlightY > -1 && highlightX < gridSize && highlightY < gridSize;
    }

    public void setTechHighlightText(String s) {
        if (!textArea.getText().equals(s)) {
            textArea.setText(formatInfoHtml(s));
            updateTechHighlight = true;
        }
    }

    public void setTechHighlight(Types.TECHNOLOGY t) {
        techHighlight = t;
    }

    private String formatInfoHtml(String body) {
        int width = Math.max(120, GUI_SIDE_PANEL_WIDTH - 24);
        return "<html><body width='" + width + "'>"
                + "<font face='serif' size='2' color='#362f24'>"
                + body
                + "</font></body></html>";
    }

    private String row(String label, Object value) {
        return "<tr><td><b><font color='#5f533e'>" + label + "</font></b></td><td>" + value + "</td></tr>";
    }

    private String formatStat(double value) {
        if (Math.rint(value) == value) {
            return Integer.toString((int) value);
        }
        return Double.toString(value);
    }

    class TribesActionListener implements ActionListener {
        int cityID;
        Vector2d position;
        ActionController ac;
        GameState gs;
        String actionType;
        Types.RESOURCE resource;
        Types.TECHNOLOGY tech;
        int unitID;

        TribesActionListener(String type) {
            this.actionType = type;
        }

        public void update(int cityID, Vector2d position, ActionController ac, GameState gs) {
            this.cityID = cityID;
            this.position = position;
            this.ac = ac;
            this.gs = gs;
        }

        public void update(Types.TECHNOLOGY t, ActionController ac, GameState gs) {
            this.tech = t;
            this.ac = ac;
            this.gs = gs;
        }

        public void update(int unitID, ActionController ac, GameState gs) {
            this.unitID = unitID;
            this.ac = ac;
            this.gs = gs;
        }

        public void setResource(Types.RESOURCE resource) {
            this.resource = resource;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            Action a = null;
            switch (actionType) {
                case "BuildRoad":
                    a = new BuildRoad(cityID);
                    ((BuildRoad) a).setPosition(position);
                    break;
                case "BurnForest":
                    a = new BurnForest(cityID);
                    ((BurnForest) a).setTargetPos(position);
                    break;
                case "ClearForest":
                    a = new ClearForest(cityID);
                    ((ClearForest) a).setTargetPos(position);
                    break;
                case "Destroy":
                    a = new Destroy(cityID);
                    ((Destroy) a).setTargetPos(position);
                    break;
                case "GrowForest":
                    a = new GrowForest(cityID);
                    ((GrowForest) a).setTargetPos(position);
                    break;
                case "ResourceGathering":
                    a = new ResourceGathering(cityID);
                    ((ResourceGathering) a).setTargetPos(position);
                    ((ResourceGathering) a).setResource(resource);
                    break;
                case "Spawn":
                    if( e.getSource() instanceof JButton) {
                        String type = ((JButton)e.getSource()).getText().split(" ")[1];
                        Types.UNIT uType = Types.UNIT.stringToType(type);
                        a = new Spawn(cityID);
                        ((Spawn) a).setTargetPos(position);
                        ((Spawn) a).setUnitType(uType);
                    }
                    break;
                case "Build":
                    if( e.getSource() instanceof JButton) {
                        String type = ((JButton)e.getSource()).getText().split(" ")[1];
                        Types.BUILDING bType = Types.BUILDING.stringToType(type);
                        a = new Build(cityID);
                        ((Build) a).setTargetPos(position);
                        ((Build) a).setBuildingType(bType);
                    }
                    break;
                case "Research":
                    if (e.getSource() instanceof JButton) {
                        a = new ResearchTech(gs.getActiveTribeID());
                        ((ResearchTech)a).setTech(tech);
                    }
                    break;
                case "HealOthers":
                    if (e.getSource() instanceof JButton) {
                        a = new HealOthers(unitID);
                    }
                    break;
                case "Disband":
                    if (e.getSource() instanceof JButton) {
                        a = new Disband(unitID);
                    }
                    break;
                case "UpgradeRammer":
                    if (e.getSource() instanceof JButton) {
                        a = new Upgrade(UPGRADE_RAMMER, unitID);
                    }
                    break;
                case "UpgradeScout":
                    if (e.getSource() instanceof JButton) {
                        a = new Upgrade(UPGRADE_SCOUT, unitID);
                    }
                    break;
                case "UpgradeBomber":
                    if (e.getSource() instanceof JButton) {
                        a = new Upgrade(UPGRADE_BOMBER, unitID);
                    }
                    break;
            }
            if (a != null) {
                ac.addAction(a, gs);
                resetHighlight();
            }
        }
    }
}
