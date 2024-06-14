# UniVaR Evaluation Scripts

### File Structure
- downstream_eval.py => Evaluation script using linear probing (cls) and kNearestNeighbour (knn) for 5 datasets (PVQRR, ValuePrism, WVS, GLOBE, and LIMA)
- downstream_eval.sh => Shell script for running various experiments using `downstream_eval.py`
- translationese_eval.py => Translationese evaluation script using linear probing (cls) and kNearestNeighbour (knn)
- translationese_eval.sh => Shell script for running various experiments using `translationese_eval.py`
- europarl_data/ => Europarl dataset split used for translationese evaluation