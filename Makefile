EXEC=bin/program
CXX=g++
CXXFLAGS=${LIB} -MMD -Wall -g
LIB=-lraylib -lbox2d
SRC=$(wildcard *.cpp)
OBJECTS=$(SRC:.cpp=.o)
DEPENDS=$(OBJECTS:.o=.d)

${EXEC}: ${OBJECTS}
	${CXX} ${CXXFLAGS} ${OBJECTS} ${LIB} -o ${EXEC}

-include ${DEPENDS}

.PHONY: clean run

clean:
	rm ${OBJECTS} ${DEPENDS} ${EXEC}

run: ${EXEC}
	@echo ~~~~~~~~~~~~~
	@${EXEC}