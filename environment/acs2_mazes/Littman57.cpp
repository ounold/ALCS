#include <vector>

#include <Maze/Mazes/Littman57.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    Littman57::Littman57() : IMazeEnvironment(
    13 * 4,
    {
        .mazeWidth = 13,
        .mazeHeight = 4,
        .goalState = {2, 9},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, PRIZE,    OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 1, 1, 1},
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            },
    }) {}
}