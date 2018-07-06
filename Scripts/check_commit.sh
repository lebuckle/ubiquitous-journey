#!/bin/bash
total=0
doxygen doxygen.cfg > /dev/null 2> >(tee check.log >&2)
cppcheck --template="{file},{line},{severity},{message}" src/ include/ > /dev/null 2> >(tee -a check.log >&2)
cpplint.py --output=vs7 $( find src include -name *.h -or -name *.cpp) > /dev/null 2> >(tee -a check.log >&2)
cd build
make > /dev/null 2> >(tee -a check.log >&2)
cd ..
errors=`wc -l < check.log`
if [ $errors \> 0 ];
then 
    echo "You have $errors errors, please fix before committing";
else
    echo "you have $errors errors, you are good to go!";
fi;
