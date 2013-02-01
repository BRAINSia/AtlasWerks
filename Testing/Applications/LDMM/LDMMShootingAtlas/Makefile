TEST_DIR:=RegressionTemp
IMAGES_DIR:=images
RUN_ATLAS:="True"
BIN_DIR:=/home/sci/jsam/code/Projects/Utah/CompOnc/trunk/build-release/bin/
TESTING_DIR:=/home/sci/jsam/code/Projects/Utah/CompOnc/trunk/Testing/

all: tests

tests: 
	export RUN_ATLAS="True";./RegressionTest.sh $(BIN_DIR) $(TESTING_DIR)

html: 
	export RUN_ATLAS="False";./RegressionTest.sh $(BIN_DIR) $(TESTING_DIR)

clean_html: 
	rm -rf $(IMAGES_DIR) index.html *.png plot.gnuplot

clean: clean_html
	rm -rf $(TEST_DIR) *~
