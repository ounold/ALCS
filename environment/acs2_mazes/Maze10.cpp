#include <vector>

#include <Maze/Mazes/Maze10.hh>
#include "Maze/EFieldState.hh"
#include "Maze/IMazeEnvironment.hh"

namespace Environments {
    using enum EFieldState;

    Maze10::Maze10() : IMazeEnvironment(
    9 * 4,
    {
        .mazeWidth = 9,
        .mazeHeight = 6,
        .goalState = {4, 3},
        .optimalAvgStepsToGoal = 4.55,
        .maze =
            {
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE },
                    { OBSTACLE, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, CORRIDOR, OBSTACLE, PRIZE,    OBSTACLE, CORRIDOR, OBSTACLE, CORRIDOR, OBSTACLE },
                    { OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE }
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1},
                    //            {1, 0, 0, 0, 0, 0, 0, 0, 1},
                    //            {1, 0, 1, 0, 1, 0, 1, 0, 1},
                    //            {1, 0, 1, 0, 1, 0, 1, 0, 1},
                    //            {1, 0, 1, 2, 1, 0, 1, 0, 1},
                    //            {1, 1, 1, 1, 1, 1, 1, 1, 1},
            },
    }) {}
}