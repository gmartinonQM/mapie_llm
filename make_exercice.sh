sed -n '/correction/!p' MAPIE_for_cosmosqa_correction.ipynb > MAPIE_for_cosmosqa.ipynb

jupyter nbconvert --clear-output --inplace MAPIE_for_cosmosqa.ipynb
