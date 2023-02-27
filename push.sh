echo Starting Commit...
echo What should the message be?
read comment
git add main.py
git add ai.py
git commit -m "$comment"
git push
