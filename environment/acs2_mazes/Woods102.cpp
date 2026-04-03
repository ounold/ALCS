#include <vector>

#include <Maze/Mazes/Woods102.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    Woods102::Woods102() : IMazeEnvironment(
    11 * 4,
    {
        .mazeWidth = 7,
        .mazeHeight = 11,
        .goalState = {1, 3},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, PRIZE,    OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, PRIZE,    OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    //            {1, 1, 1, 1, 1, 1, 1},
                    //            {1, 0, 1, 2, 1, 0, 1},
                    //            {1, 0, 1, 0, 1, 0, 0},
                    //            {1, 0, 0, 0, 0, 0, 0},
                    //            {1, 0, 1, 0, 1, 0, 1},
                    //            {1, 1, 1, 1, 1, 1, 0},
                    //            {1, 0, 1, 0, 1, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 1, 0, 1, 0, 1},
                    //            {1, 0, 1, 2, 1, 0, 1},
                    //            {1, 1, 1, 1, 1, 1, 1},
            },
    }) {}
}