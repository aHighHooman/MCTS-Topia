package gui;

import core.actions.Action;

import javax.swing.*;
import javax.swing.text.DefaultCaret;
import java.awt.*;

import static core.Constants.GUI_SIDE_PANEL_WIDTH;
import static core.Constants.GUI_TECH_PANEL_HEIGHT;

public class GameLogView extends JComponent {

    private final Dimension size;
    private final JTextArea textArea;

    GameLogView() {
        this.size = new Dimension(GUI_SIDE_PANEL_WIDTH, GUI_TECH_PANEL_HEIGHT);

        textArea = new JTextArea("No events yet.");
        textArea.setEditable(false);
        textArea.setBackground(new Color(252, 248, 236));
        textArea.setBorder(BorderFactory.createEmptyBorder(4, 6, 4, 6));
        textArea.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 11));
        textArea.setLineWrap(true);
        textArea.setWrapStyleWord(true);
        DefaultCaret caret = (DefaultCaret) textArea.getCaret();
        caret.setUpdatePolicy(DefaultCaret.ALWAYS_UPDATE);

        JScrollPane scrollPane = new JScrollPane(textArea);
        scrollPane.setBorder(BorderFactory.createEmptyBorder());
        scrollPane.setPreferredSize(size);

        setLayout(new BorderLayout());
        add(scrollPane, BorderLayout.CENTER);
    }

    void addAction(int turn, int playerId, Action action) {
        if (action != null) {
            addEntry("Turn " + turn + ", Player " + playerId + ": " + action);
        }
    }

    void addEntry(String entry) {
        if ("No events yet.".equals(textArea.getText())) {
            textArea.setText("");
        }
        textArea.append(entry + System.lineSeparator());
    }

    public Dimension getPreferredSize() {
        return size;
    }
}
