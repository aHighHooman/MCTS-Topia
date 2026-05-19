package core;

import core.game.Board;
import org.json.JSONArray;
import org.json.JSONObject;

import static core.TribesConfig.ALLEGIANCE_MAX;

public class Diplomacy {

    // stores the allegiances of the tribes
    private int[][] allegianceStatus;
    private Types.RELATIONSHIP[][] relationshipStatus;
    private int[][] lastRelationChangeTurn;
    private int[][] pendingTreatyOffers;
    private Types.RELATIONSHIP[][] pendingOfferTypes;

    /**
     * Creates allegiances for each of the existing tribes
     *
     * @param size number of tribes in-game
     */
    public Diplomacy(int size) {
        this.allegianceStatus = new int[size][size];
        this.relationshipStatus = new Types.RELATIONSHIP[size][size];
        this.lastRelationChangeTurn = new int[size][size];
        this.pendingTreatyOffers = new int[size][size];
        this.pendingOfferTypes = new Types.RELATIONSHIP[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                relationshipStatus[i][j] = Types.RELATIONSHIP.PEACE;
                lastRelationChangeTurn[i][j] = -1;
                pendingTreatyOffers[i][j] = -1;
                pendingOfferTypes[i][j] = null;
            }
        }
    }

    public Diplomacy(JSONObject diplomacyInfo, int size) {
        this(size);

        if (diplomacyInfo.has("allegiance")) {
            JSONArray jAllegiance = diplomacyInfo.getJSONArray("allegiance");
            for (int i = 0; i < jAllegiance.length() && i < size; i++) {
                JSONArray row = jAllegiance.getJSONArray(i);
                for (int j = 0; j < row.length() && j < size; j++) {
                    allegianceStatus[i][j] = row.getInt(j);
                }
            }
        }

        if (diplomacyInfo.has("relationship")) {
            JSONArray jRelationships = diplomacyInfo.getJSONArray("relationship");
            for (int i = 0; i < jRelationships.length() && i < size; i++) {
                JSONArray row = jRelationships.getJSONArray(i);
                for (int j = 0; j < row.length() && j < size; j++) {
                    Types.RELATIONSHIP relationship = Types.RELATIONSHIP.getTypeByKey(row.getInt(j));
                    if (relationship != null) {
                        relationshipStatus[i][j] = relationship;
                    }
                }
            }
        } else {
            syncAllRelationshipsFromAllegiance();
        }

        if (diplomacyInfo.has("lastRelationChangeTurn")) {
            JSONArray jLastChanges = diplomacyInfo.getJSONArray("lastRelationChangeTurn");
            for (int i = 0; i < jLastChanges.length() && i < size; i++) {
                JSONArray row = jLastChanges.getJSONArray(i);
                for (int j = 0; j < row.length() && j < size; j++) {
                    lastRelationChangeTurn[i][j] = row.getInt(j);
                }
            }
        }

        if (diplomacyInfo.has("pendingTreatyOffers")) {
            JSONArray jPendingOffers = diplomacyInfo.getJSONArray("pendingTreatyOffers");
            for (int i = 0; i < jPendingOffers.length() && i < size; i++) {
                JSONArray row = jPendingOffers.getJSONArray(i);
                for (int j = 0; j < row.length() && j < size; j++) {
                    pendingTreatyOffers[i][j] = row.getInt(j);
                }
            }
        }

        if (diplomacyInfo.has("pendingOfferTypes")) {
            JSONArray jPendingOfferTypes = diplomacyInfo.getJSONArray("pendingOfferTypes");
            for (int i = 0; i < jPendingOfferTypes.length() && i < size; i++) {
                JSONArray row = jPendingOfferTypes.getJSONArray(i);
                for (int j = 0; j < row.length() && j < size; j++) {
                    int key = row.getInt(j);
                    pendingOfferTypes[i][j] = key == -1 ? null : Types.RELATIONSHIP.getTypeByKey(key);
                }
            }
        }
    }

    public int[][] getAllegianceStatus() {
        return allegianceStatus;
    }

    public void setAllegianceStatus(int x, int y, int val) {
        allegianceStatus[x][y] = val;
    }

    public Types.RELATIONSHIP[][] getRelationshipStatus() {
        return relationshipStatus;
    }

    public Types.RELATIONSHIP[][] copyRelationshipStatus() {
        Types.RELATIONSHIP[][] copy = new Types.RELATIONSHIP[relationshipStatus.length][relationshipStatus.length];
        for (int i = 0; i < relationshipStatus.length; i++) {
            System.arraycopy(relationshipStatus[i], 0, copy[i], 0, relationshipStatus[i].length);
        }
        return copy;
    }

    public Types.RELATIONSHIP getRelationship(int x, int y) {
        return relationshipStatus[x][y];
    }

    public void setRelationship(int x, int y, Types.RELATIONSHIP relationship) {
        setRelationship(x, y, relationship, -1);
    }

    public void setRelationship(int x, int y, Types.RELATIONSHIP relationship, int turn) {
        relationshipStatus[x][y] = relationship;
        relationshipStatus[y][x] = relationship;
        if (turn >= 0) {
            lastRelationChangeTurn[x][y] = turn;
            lastRelationChangeTurn[y][x] = turn;
        }
    }

    public int getLastRelationChangeTurn(int x, int y) {
        return lastRelationChangeTurn[x][y];
    }

    public int getPendingTreatyOffer(int x, int y) {
        return pendingTreatyOffers[x][y];
    }

    public void setPendingTreatyOffer(int x, int y, int proposerTribeId) {
        pendingTreatyOffers[x][y] = proposerTribeId;
    }

    public Types.RELATIONSHIP getPendingOfferType(int x, int y) {
        return pendingOfferTypes[x][y];
    }

    public void setPendingOfferType(int x, int y, Types.RELATIONSHIP relationship) {
        pendingOfferTypes[x][y] = relationship;
    }

    public boolean hasPendingOffer(int proposerId, int targetId, Types.RELATIONSHIP relationship) {
        return pendingTreatyOffers[proposerId][targetId] == proposerId && pendingOfferTypes[proposerId][targetId] == relationship;
    }

    public boolean hasPendingOfferBetween(int tribeA, int tribeB) {
        return pendingOfferTypes[tribeA][tribeB] != null || pendingOfferTypes[tribeB][tribeA] != null;
    }

    public void proposeRelationship(int proposerId, int targetId, Types.RELATIONSHIP relationship) {
        pendingTreatyOffers[proposerId][targetId] = proposerId;
        pendingOfferTypes[proposerId][targetId] = relationship;
    }

    public void clearPendingOffer(int proposerId, int targetId) {
        pendingTreatyOffers[proposerId][targetId] = -1;
        pendingOfferTypes[proposerId][targetId] = null;
    }

    public void clearPendingOffersBetween(int tribeA, int tribeB) {
        clearPendingOffer(tribeA, tribeB);
        clearPendingOffer(tribeB, tribeA);
    }

    public void acceptPendingRelationship(int proposerId, int targetId, int currentTurn) {
        Types.RELATIONSHIP relationship = pendingOfferTypes[proposerId][targetId];
        if (relationship != null) {
            setRelationship(proposerId, targetId, relationship, currentTurn);
            clearPendingOffersBetween(proposerId, targetId);
        }
    }

    public void cancelTreaty(int tribeA, int tribeB, int currentTurn) {
        setRelationship(tribeA, tribeB, Types.RELATIONSHIP.PEACE, currentTurn);
        clearPendingOffersBetween(tribeA, tribeB);
    }

    public boolean isAtWar(int x, int y) {
        return getRelationship(x, y) == Types.RELATIONSHIP.WAR;
    }

    public boolean isTreaty(int x, int y) {
        return getRelationship(x, y) == Types.RELATIONSHIP.TREATY;
    }

    /**
     * Creates allegiances for each of the existing tribes
     *
     * @param value         value the allegiance will change by, will be negative if called by an action, positive if called by checkConsequences()
     * @param initTribeID   ID for the tribe initiating the action
     * @param targetTribeID ID for the target tribe of the action
     */
    public void updateAllegiance(int value, int initTribeID, int targetTribeID) {
        updateAllegiance(value, initTribeID, targetTribeID, -1);
    }

    public void updateAllegiance(int value, int initTribeID, int targetTribeID, int currentTurn) {
        // checks if increase or decreasing the allegiance will go over the limit
        // if so, set the value to the the difference of the absoluteMax and the current allegiance value
        if ((this.allegianceStatus[initTribeID][targetTribeID] + value < -ALLEGIANCE_MAX) || (this.allegianceStatus[initTribeID][targetTribeID] + value > ALLEGIANCE_MAX)) {
            value = Integer.signum(value) * (ALLEGIANCE_MAX - Math.abs(this.allegianceStatus[initTribeID][targetTribeID]));
        }
        // adds the value to the allegiance
        this.allegianceStatus[initTribeID][targetTribeID] = this.allegianceStatus[initTribeID][targetTribeID] + value;
        this.allegianceStatus[targetTribeID][initTribeID] = this.allegianceStatus[targetTribeID][initTribeID] + value;
        syncRelationshipFromAllegiance(initTribeID, targetTribeID, currentTurn);
    }

    /**
     * Checks the consequences of the change in relationship, to see if any more allegiances need to be updated
     *
     * @param value         value the allegiance will change by
     * @param initTribeID   ID for the tribe initiating the action
     * @param targetTribeID ID for the target tribe of the action
     */
    public void checkConsequences(int value, int initTribeID, int targetTribeID) {
        checkConsequences(value, initTribeID, targetTribeID, -1);
    }

    public void checkConsequences(int value, int initTribeID, int targetTribeID, int currentTurn) {
        // inverting and halving the value
        value = value / -2;
        // checks all relationships for the target tribe
        for (int i = 0; i < allegianceStatus.length; i++) {
            // if the target tribe has a negative relationship and it is not with the initiating tribe
            if ((allegianceStatus[i][targetTribeID]) < 0 && (i != initTribeID)) {
                // call update allegiance for the target tribe
                updateAllegiance(value, i, initTribeID, currentTurn);
            }
        }
    }

    private void syncRelationshipFromAllegiance(int initTribeID, int targetTribeID, int currentTurn) {
        if (relationshipStatus[initTribeID][targetTribeID] == Types.RELATIONSHIP.TREATY ||
                relationshipStatus[initTribeID][targetTribeID] == Types.RELATIONSHIP.WAR) {
            return;
        }

        Types.RELATIONSHIP nextRelationship =
                allegianceStatus[initTribeID][targetTribeID] <= -(int)(ALLEGIANCE_MAX / 2.0f)
                        ? Types.RELATIONSHIP.WAR
                        : Types.RELATIONSHIP.PEACE;

        if (relationshipStatus[initTribeID][targetTribeID] != nextRelationship) {
            setRelationship(initTribeID, targetTribeID, nextRelationship, currentTurn);
            clearPendingOffersBetween(initTribeID, targetTribeID);
        }
    }

    private void syncAllRelationshipsFromAllegiance() {
        for (int i = 0; i < allegianceStatus.length; i++) {
            for (int j = i + 1; j < allegianceStatus.length; j++) {
                syncRelationshipFromAllegiance(i, j, -1);
            }
        }
    }

    public JSONObject toJSON() {
        JSONObject diplomacyInfo = new JSONObject();
        diplomacyInfo.put("allegiance", matrixToJson(allegianceStatus));
        diplomacyInfo.put("relationship", relationshipMatrixToJson(relationshipStatus));
        diplomacyInfo.put("lastRelationChangeTurn", matrixToJson(lastRelationChangeTurn));
        diplomacyInfo.put("pendingTreatyOffers", matrixToJson(pendingTreatyOffers));
        diplomacyInfo.put("pendingOfferTypes", nullableRelationshipMatrixToJson(pendingOfferTypes));
        return diplomacyInfo;
    }

    private JSONArray matrixToJson(int[][] matrix) {
        JSONArray matrixJson = new JSONArray();
        for (int[] row : matrix) {
            JSONArray rowJson = new JSONArray();
            for (int value : row) {
                rowJson.put(value);
            }
            matrixJson.put(rowJson);
        }
        return matrixJson;
    }

    private JSONArray relationshipMatrixToJson(Types.RELATIONSHIP[][] matrix) {
        JSONArray matrixJson = new JSONArray();
        for (Types.RELATIONSHIP[] row : matrix) {
            JSONArray rowJson = new JSONArray();
            for (Types.RELATIONSHIP relationship : row) {
                rowJson.put(relationship.getKey());
            }
            matrixJson.put(rowJson);
        }
        return matrixJson;
    }

    private JSONArray nullableRelationshipMatrixToJson(Types.RELATIONSHIP[][] matrix) {
        JSONArray matrixJson = new JSONArray();
        for (Types.RELATIONSHIP[] row : matrix) {
            JSONArray rowJson = new JSONArray();
            for (Types.RELATIONSHIP relationship : row) {
                rowJson.put(relationship == null ? -1 : relationship.getKey());
            }
            matrixJson.put(rowJson);
        }
        return matrixJson;
    }

    /**
     * Creates a copy of this Diplomacy object
     * @return another instance, copy of this.
     */
    public Diplomacy copy()
    {
        Diplomacy copyD = new Diplomacy(this.allegianceStatus.length);
        for(int i =0; i < getAllegianceStatus().length; ++i)
        {
            for (int j=0; j < getAllegianceStatus()[i].length; ++j) {
                copyD.setAllegianceStatus(i, j, getAllegianceStatus()[i][j]);
                copyD.relationshipStatus[i][j] = relationshipStatus[i][j];
                copyD.lastRelationChangeTurn[i][j] = lastRelationChangeTurn[i][j];
                copyD.pendingTreatyOffers[i][j] = pendingTreatyOffers[i][j];
                copyD.pendingOfferTypes[i][j] = pendingOfferTypes[i][j];
            }
        }
        return copyD;
    }

    //Method to test the diplomacy is working
    public void logAllegiance(Board b) {
        System.out.println("NEW LOG");
        for (int i = 0; i < allegianceStatus.length; i++) {
            System.out.print(b.getTribes()[i].getName() + ": ");
            for (int j = 0; j < allegianceStatus.length; j++) {
                System.out.print(this.allegianceStatus[i][j] + "/" + relationshipStatus[i][j] + ", ");
            }
            System.out.println();
        }
    }
}
