#!/bin/bash

# Run your tests here
cd test
make sinUnitTest
make sinRegrTest
make runsinUnitTest
make runsinRegrTest
make clean_sinUnit
make clean_sinRegr

# Capture the exit status of the tests
status=$?

# If tests failed, exit with an error
[ $status -ne 0 ] && exit $status

# Otherwise, exit with success
exit 0
