#include <vector>

#include <Maze/Mazes/Woods100.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    Woods100::Woods100() : IMazeEnvironment(
    9 * 4,
    {
        .mazeWidth = 9,
        .mazeHeight = 3,
        .goalState = {1, 4},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, PRIZE,    CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 2, 0, 0, 0, 1},
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1},
            },
    }) {}
}