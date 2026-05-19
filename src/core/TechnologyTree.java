package core;

import org.json.JSONArray;
import org.json.JSONObject;
import java.util.ArrayList;
import java.util.Random;

public class TechnologyTree {

    // Array that indicates if tech. idx is researched already.
    private boolean[] researched;

    // Flag that indicates if the whole tree has been researched.
    private boolean everythingResearched = false;

    /**
     * Creates a TechnologyTree.
     */
    public TechnologyTree(){
        researched = new boolean[Types.TECHNOLOGY.values().length];
//        Arrays.fill(researched, true); everythingResearched = true; //Buff for debug purposes, keep this commented for real games.
    }

    /**
     * Creates a TechnologyTree with a starting set of technologies researched
     * @param researched array that indicates which technologies are researched already.
     */
    public TechnologyTree(boolean[] researched){
        this.researched = new boolean[Types.TECHNOLOGY.values().length];
        System.arraycopy(researched, 0, this.researched, 0, Math.min(researched.length, this.researched.length));
        checkEverythingResearched();
    }

    /**
     * Creates a new TechTree from a JSON object
     * @param jTechnologyTree technology tree object to load
     */
    public TechnologyTree(JSONObject jTechnologyTree){
        researched = new boolean[Types.TECHNOLOGY.values().length];

        if (jTechnologyTree.has("researchedTechIds")) {
            JSONArray jResearchedTechIds = jTechnologyTree.getJSONArray("researchedTechIds");
            for (int i = 0; i < jResearchedTechIds.length(); i++) {
                Types.TECHNOLOGY technology = Types.TECHNOLOGY.getTypeById(jResearchedTechIds.getString(i));
                if (technology != null) {
                    researched[technology.ordinal()] = true;
                }
            }
        } else {
            JSONArray jResearched = jTechnologyTree.getJSONArray("researched");
            for (int i=0; i<jResearched.length() && i<researched.length; i++){
                researched[i] = jResearched.getBoolean(i);
            }
        }

        if (jTechnologyTree.has("everythingResearched")) {
            everythingResearched = jTechnologyTree.getBoolean("everythingResearched");
        } else {
            checkEverythingResearched();
        }
    }

    /**
     * Checks if a technology is researched.
     * @param target technology to check if it's researched.
     * @return true if the tech is researched.
     */
    public boolean isResearched(Types.TECHNOLOGY target) {
        if (target != null) {
            return researched[target.ordinal()];
        }
        return false;
    }

    /**
     * Copies this tree and returns it.
     * @return a copy of this tree.
     */
    public TechnologyTree copy(){
        return new TechnologyTree(researched);
    }

    /**
     * Checks if a technology is researchable.
     * It won't be if it's already researched or its requirement(s) are not met yet.
     * @param target technology to query
     * @return true if target can be researched in this tree.
     */
    public boolean isResearchable(Types.TECHNOLOGY target) {
        Types.TECHNOLOGY requirement = target.getParentTech();
        if (isResearched(target)) {
            return false;
        }
        if (requirement == null || isResearched(requirement)) {
            return true;
        }
        for (Types.TECHNOLOGY child : target.getChildTech()) {
            if (isResearched(child)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Computes if every technology is already researched in this tree.
     */
    private void checkEverythingResearched(){
        for (boolean b : researched){
            if(!b){return;}
        }
        everythingResearched = true;
    }

    /**
     * Returns if this tree is fully researched.
     * @return true if this tree is fully researched.
     */
    public boolean isEverythingResearched(){
        return everythingResearched;
    }

    /**
     * Performs a research of a technology
     * @param target technology to be researched.
     * @return true if it could be researched.
     */
    public boolean doResearch(Types.TECHNOLOGY target) {
        if(isResearchable(target)) {
            researched[target.ordinal()] = true;
            if(target.getChildTech().isEmpty()) {
                checkEverythingResearched();
            }
            return true;
        }
        return false;
   }

    /**
     * Researches the initial technology. Allows for research of techs with parents' that aren't researched
     * @param target the target tech.
     */
   public void doResearchInit(Types.TECHNOLOGY target){
        researched[target.ordinal()] = true;
   }

    /**
     * Researches a technology at random within the tree.
     * @param rnd random generator to pick a tech.
     * @return true if a technology could be researched.
     */
   public boolean researchAtRandom(Random rnd)
   {
       if(isEverythingResearched()) return false;

       ArrayList<Types.TECHNOLOGY> available = new ArrayList<>();
       for(Types.TECHNOLOGY tech : Types.TECHNOLOGY.values())
       {
           if(isResearchable(tech))
               available.add(tech);
       }

       if(available.size() == 0)
           return false;

       Types.TECHNOLOGY t = available.get(rnd.nextInt(available.size()));
       return doResearch(t);
   }

    /**
     * Returns the list of researched technologies
     * @return boolean array with the researched technologies.
     */
    public boolean[] getResearched() {
        return researched;
    }

    public JSONArray getResearchedTechIds() {
        JSONArray researchedTechIds = new JSONArray();
        for (Types.TECHNOLOGY technology : Types.TECHNOLOGY.values()) {
            if (isResearched(technology)) {
                researchedTechIds.put(technology.getId());
            }
        }
        return researchedTechIds;
    }

    /**
     * Returns the number of technologies researched
     * @return the number of technologies researched
     */
    public int getNumResearched()
    {
        int count = 0;
        for(Types.TECHNOLOGY tech : Types.TECHNOLOGY.values())
        {
            if(isResearched(tech))
                count++;
        }
        return count;
    }
}
