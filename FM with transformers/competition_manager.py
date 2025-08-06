from dataclasses import dataclass
from dataclasses import field
from typing import List
from sklearn.preprocessing import OneHotEncoder

@dataclass
class Team:
    players: List[int] = field(default_factory=list)  #field(default="Push-up")


from datetime import date


@dataclass
class Competition:
    team_a: Team #= field(default_factory=Team)
    team_b: Team #= field(default_factory=Team)
    score_diff: int
    date: date #= field(default_factory=lambda: date.today())
    scores_stats: dict = field(default_factory=dict)

    def __calculate_score(comp, pl_ratings):
        scores_a = sum([pl_ratings[pl] for pl in comp.team_a])
        scores_b = sum([pl_ratings[pl] for pl in comp.team_b])
        if set([1,19]) <= set(comp.team_a):
            scores_a += 5
        if set([1,19]) <= set(comp.team_b):
            scores_b += 5
        return 50 - abs(scores_a - scores_b)

    def score(self, ratings):
        score = self.__calculate_score(ratings)
        return ScoredCompetition(ratings = ratings, score = score, competition = self)

    def explain(self, players, ratings):
        return ([players[pl] for pl in self.team_a], [players[pl] for pl in self.team_b]) 
        

@dataclass
class ScoredCompetition:
    ratings:dict
    score: float
    competition: Competition

class CompetitionManager():

    def __init__(self, players, ratings):
        self._players = players
        self._ratings = ratings

    def generate_competitions(self, n=100):
        exp_competitions = generate_random_competitions(n, self._players)
        return exp_competitions

    def score(self, competitions):
        scored = [comp.score(self._ratings) for comp in competitions]
        return scored

    def convert_to_df(self, scored_competitions):
        def competition_as_ohe_features(competition, debug = False):
            competition_feat = np.full([1, feat_space_unary], "-")
            team_a = competition.competition.team_a
            team_b = competition.competition.team_b
            if debug:
                print(f"Team A: {team_a}    Team B: {team_b} ")
        
            # print(f"Populating values from team_a: {team_a}")
            for comb_a in team_a:
                competition_feat[0][comb_a -1] = "a" # '0' as we create one dimensional feature vector on every iteration and then concatenate
        
            # if debug:
            #     print(f"competition_feat after team A: {competition_feat}")
        
            # combs_b = list(itertools.combinations(team_b, 2))
            # print(f"Team B. Number of combs {len(combs_b)} ") #Combinations: {combs_b}
        
            # print(f"Populating values from team_b: {team_b}")
            for comb_b in team_b:
                competition_feat[0][comb_b - 1] = "b" # '0' as we create one dimensional feature vector on every iteration and then concatenate

            if debug:
                print(f"Current features {competition_feat}")
            return competition_feat
    
        competitions_feat = np.full([1, feat_space_unary], "-")
        comps_scores = []
        
        for competition in scored_competitions:
            competition_feat = competition_as_ohe_features(competition, debug = False)
            competitions_feat = np.concatenate([competitions_feat, competition_feat])
            comps_scores.append(competition.score)

        competitions_feat = np.delete(competitions_feat, (0), axis=0)
        competitions_df = pd.DataFrame.from_records(competitions_feat)
        scores_df = pd.DataFrame(comps_scores, columns=['score'])

        return (competitions_df, scores_df)

    def convert_to_pair_featured(self, scored_competitions):

        import itertools
        combs = itertools.combinations(team_members_with_ids, 2)

        input_features = {}
        inverted_idx_input_features = {}
        for idx, combination in enumerate(combs):
            input_features[idx] = combination
            # print(f"{idx} {combination} : {team_members_with_ids[combination[0]]} & {team_members_with_ids[combination[1]]}")
            inverted_idx_input_features[combination] = idx

        
        feat_space_pair = len(inverted_idx_input_features)
        
        def tint(tupple):
            return (int(tupple[0]), int(tupple[1]))
        
        def comp_to_pair_feat(competition, debug = False):
            competition_feat = np.full([1, feat_space_pair], "-")
            team_a = competition.competition.team_a
            team_b = competition.competition.team_b
            # print(f"Team A: {team_a}")
            # print(f"Team B: {team_b}")
            combs_a = list(itertools.combinations(team_a, 2))
            # print(f"Team A. Number of combs {len(combs_a)}.") # Combinations: {combs_a}
            
            for comb_a in combs_a:
                feat_idx = inverted_idx_input_features.get((comb_a[0].item(), comb_a[1].item())) or inverted_idx_input_features.get((comb_a[1].item(), comb_a[0].item()))
                if debug:
                    print(f"Index for a pair {tint(comb_a)} is {feat_idx}")
                competition_feat[0][feat_idx] = "a"
        
            if debug:
                print(f"competition_feat after team A: {competition_feat}")
        
            combs_b = list(itertools.combinations(team_b, 2))
            # print(f"Team B. Number of combs {len(combs_b)} ") #Combinations: {combs_b}
            
            for comb_b in combs_b:
                feat_idx = inverted_idx_input_features.get((comb_b[0].item(), comb_b[1].item())) or inverted_idx_input_features.get((comb_b[1].item(), comb_b[0].item()))
                if debug:
                    print(f"Index for a pair {tint(comb_b)} is {feat_idx}")
                competition_feat[0][feat_idx] = "b"   # we changed -1 into +1 as we focus on pairs and not on opposite teams
                # print(f"competition_feat after comb_b_iter: {competition_feat}")
                
        
            # if debug:
            #     print(f"Current features {competition_feat}")
            return competition_feat

        competitions_feat = np.full([1, feat_space_pair], "-")
        comps_scores = []
        
        for competition in scored_competitions:
            competition_feat = comp_to_pair_feat(competition, debug = False)
            competitions_feat = np.concatenate([competitions_feat, competition_feat])
            comps_scores.append(competition.score)

        competitions_feat = np.delete(competitions_feat, (0), axis=0)
        competitions_df = pd.DataFrame.from_records(competitions_feat)
        scores_df = pd.DataFrame(comps_scores, columns=['score'])

        return (competitions_df, scores_df)
        

    def ohe(self, competitions_df):
        drop_enc = OneHotEncoder(drop='first', handle_unknown='ignore').fit(competitions_df)
        drop_enc.categories_
        
        final_df = drop_enc.transform(competitions_df).toarray()
        final_df = pd.DataFrame(final_df, columns=drop_enc.get_feature_names_out())
        print(final_df.shape)
        return final_df

    def explain_competitions(self, competitions, indexes):
        comps = [ competitions[idx] for idx in indexes ] 
        print(comps)
        return [comp.explain(self._players, self._ratings) for comp in comps ]
        