#include <vector>

#include <Maze/Mazes/MazeE2.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    MazeE2::MazeE2() : IMazeEnvironment(
    9 * 4,
    {
        .mazeWidth = 9,
        .mazeHeight = 9,
        .goalState = {4, 4},
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, PRIZE,    CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE }
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 2, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1}
            },
    }) {}
}