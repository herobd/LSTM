#Brian Davis
#make file for RNN project

MAIN_OBJ_FILES = obj/LSTM.o obj/ArffReader.o obj/Attribute.o #obj/DiscreteAttribute.o obj/ContinuousAttribute.o
CFLAGS = -c -Wall -I . -O3 -std=c++0x
#CFLAGS = -c -g -Wall -I . -O0 -std=c++0x


bin: bin/LSTM


clean:
	- rm -f bin/LSTM
	- rm -f obj/*.o

	

#executable dependencies

bin/LSTM: obj/main.o $(MAIN_OBJ_FILES)
	g++ obj/main.o $(MAIN_OBJ_FILES) -o bin/LSTM

#object file dependencies

obj/main.o:  ./main.cpp ./LSTM.h
	 g++ $(CFLAGS) -o obj/main.o  ./main.cpp

obj/ArffReader.o: ./ArffReader.cpp ./ArffReader.h
		g++ $(CFLAGS) -o obj/ArffReader.o ./ArffReader.cpp

obj/LSTM.o: ./LSTM.cpp ./LSTM.h
	g++ $(CFLAGS) -o obj/LSTM.o ./LSTM.cpp
	
obj/Attribute.o: ./Attribute.h ./Attribute.cpp
	g++ $(CFLAGS) -o obj/Attribute.o ./Attribute.cpp
	
obj/ContinuousAttribute.o: ./ContinuousAttribute.h ./ContinuousAttribute.cpp
	g++ $(CFLAGS) -o obj/ContinuousAttribute.o ./ContinuousAttribute.cpp
	
obj/DiscreteAttribute.o: ./DiscreteAttribute.h ./DiscreteAttribute.cpp
	g++ $(CFLAGS) -o obj/DiscreteAttribute.o ./DiscreteAttribute.cpp
	
obj/AttributeKey.o: ./AttributeKey.h ./AttributeKey.cpp
	g++ $(CFLAGS) -o obj/AttributeKey.o ./AttributeKey.cpp
	
	


