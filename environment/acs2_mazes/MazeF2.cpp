#include <vector>

#include <Maze/Mazes/MazeF2.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    MazeF2::MazeF2() : IMazeEnvironment(
    6 * 4,
    {
        .mazeWidth = 5,
        .mazeHeight = 6,
        .goalState = {1, 3},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, PRIZE,    OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE }
                    //            {1, 1, 1, 1, 1},
                    //            {1, 0, 0, 2, 1},
                    //            {1, 0, 1, 1, 1},
                    //            {1, 0, 0, 1, 1},
                    //            {1, 0, 1, 1, 1},
                    //            {1, 1, 1, 1, 1},
            },
    }) {}
}