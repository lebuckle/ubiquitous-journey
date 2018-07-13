# Run pylint and print results to pylint.txt
pylint *.py 2>&1 > pylint.txt
# Take the last line
output=$(tail -2 pylint.txt)
echo $output
