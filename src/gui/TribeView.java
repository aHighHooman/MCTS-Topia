package gui;

import core.Types;
import core.actors.Tribe;
import core.game.Game;
import core.game.GameState;
import players.Agent;

import javax.swing.*;
import javax.swing.text.DefaultCaret;
import java.awt.*;
import java.util.Arrays;
import java.util.Comparator;

import static core.Constants.GUI_SIDE_PANEL_WIDTH;
import static core.Constants.GUI_TECH_PANEL_HEIGHT;


public class TribeView extends JComponent {

    private Dimension size;
    private JEditorPane textArea;
    private GameState gs;
    private Game game;
    private String lastHtml = "";

    TribeView(Game game)
    {
        this.size = new Dimension(GUI_SIDE_PANEL_WIDTH, GUI_TECH_PANEL_HEIGHT);
        this.game = game;

        textArea = new JEditorPane("text/html", "");
        textArea.setPreferredSize(this.size);
        Font textFont = new Font(textArea.getFont().getName(), Font.PLAIN, 12);
        textArea.setFont(textFont);
        textArea.setEditable(false);
        textArea.setBackground(Color.lightGray);
        DefaultCaret caret = (DefaultCaret)textArea.getCaret();
        caret.setUpdatePolicy(DefaultCaret.NEVER_UPDATE);

        JScrollPane scrollPane = new JScrollPane(textArea);
        scrollPane.setBorder(BorderFactory.createEmptyBorder());
        scrollPane.setPreferredSize(this.size);

        this.setLayout(new BorderLayout());
        this.add(scrollPane, BorderLayout.CENTER);
    }

    public void paintComponent(Graphics gx)
    {
        super.paintComponent(gx);
    }

    private void updateContents()
    {
        if (gs != null) {
            Tribe[] tribes = gs.getTribes().clone();

            Tribe t_this = tribes[gs.getActiveTribeID()];
            StringBuilder s = new StringBuilder("<html><body style='font-family:sans-serif;font-size:11px;'>");
            s.append("<p><b>").append(t_this.getName()).append("</b> acting<br/>")
                    .append("Score: ").append(t_this.getScore())
                    .append(" | Stars: ").append(t_this.getStars())
                    .append(" (+").append(t_this.getMaxProduction(gs)).append(")")
                    .append("</p><hr><h2 style='margin:4px 0;'>Rankings</h2>");
            s.append("<table cellpadding='2' cellspacing='0'>")
                    .append("<tr><th align='left'>Player</th><th>Stars</th><th>Score</th><th>Cities</th><th>Techs</th><th>SPT</th></tr>");
            Agent[] agents = game.getPlayers();

            Arrays.sort(tribes, Comparator.comparing(Tribe::getReverseScore));
            for (Tribe t : tribes) {
                Agent ag = agents[t.getTribeId()];
                Types.RESULT winState = t.getWinner();
                String w = "";
                String[] agentChunks = ag.getClass().toString().split("\\.");
                String agentName = agentChunks[agentChunks.length - 1];

                if (winState != Types.RESULT.INCOMPLETE) w = " (" + winState.toString() + ")";
                s.append("<tr><td>")
                        .append(t.getTribeId()).append(" ")
                        .append(t.getName()).append(" ")
                        .append(agentName).append(w)
                        .append("</td><td align='right'>")
                        .append(t.getStars())
                        .append("</td><td align='right'>")
                        .append(t.getScore())
                        .append("</td><td align='right'>")
                        .append(t.getNumCities())
                        .append("</td><td align='right'>")
                        .append(t.getTechTree().getNumResearched())
                        .append("</td><td align='right'>")
                        .append(t.getMaxProduction(gs))
                        .append("</td></tr>");
            }
            s.append("</table></body></html>");

            String html = s.toString();
            if (!lastHtml.equals(html)) {
                lastHtml = html;
                textArea.setText(html);
            }
        }
    }


    /**

     */
    void refresh(GameState gameState)
    {
        this.gs = gameState;
        updateContents();
    }


    /**
     * Gets the dimensions of the window.
     * @return the dimensions of the window.
     */
    public Dimension getPreferredSize() {
        return size;
    }

}
