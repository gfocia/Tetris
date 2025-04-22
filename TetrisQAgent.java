package src.pas.tetris.agents;

import java.util.List;
import java.util.Random;
import java.util.Iterator; 

import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Block;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer 
import edu.bu.tetris.nn.layers.ReLU; // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;
import edu.bu.tetris.utils.Coordinate; 

public class TetrisQAgent extends QAgent {

    public static final double EXPLORATION_PROB = 0.05;
    private Random random;

    public TetrisQAgent(String name) {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction() {
         System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector

        final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        final int inputSize = 5; // Feature size
        final int hiddenDim = (int)Math.pow(inputSize, 2); // Hidden layer size
        final int outDim = 1; // Single Q-value output

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputSize, hiddenDim));
        qFunction.add(new ReLU()); // ReLU activation for non-linearity
        qFunction.add(new Dense(hiddenDim, outDim));
        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */

    @Override
    public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
        //System.out.println("GetQFunction has been called");
        return calculateSimplifiedFeatures(game, potentialAction);
    }

    private Matrix calculateSimplifiedFeatures(final GameView game, final Mino potentialAction) {
        Matrix featureMatrix = Matrix.zeros(1, 5);
    
        Matrix grayscaleMatrix = null;
    
        // Try to fetch the grayscale matrix representation of the board
        try {
            grayscaleMatrix = game.getGrayscaleImage(potentialAction);
        } catch (Exception e) {
            System.err.println("[ERROR] Unable to fetch grayscale matrix!");
            e.printStackTrace();
            System.exit(-1); // Exit if the grayscale matrix cannot be fetched
        }
    
        // Calculate features using helper methods and grayscale matrix
        int baseHeight = calculateBaseHeightFromGrayscale(grayscaleMatrix);
        int bumpiness = calculateBumpinessFromGrayscale(grayscaleMatrix);
        int emptyBelow = calculateEmptySpacesFromGrayscale(grayscaleMatrix);
        int fullRows = calculateFullRowsFromGrayscale(grayscaleMatrix);
    
        // Determine Mino type using try-catch to handle unexpected values
        int minoType = -1;
        try {
            switch (potentialAction.getType()) {
                case I:
                    minoType = 0;
                    break;
                case J:
                    minoType = 1;
                    break;
                case L:
                    minoType = 2;
                    break;
                case O:
                    minoType = 3;
                    break;
                case S:
                    minoType = 4;
                    break;
                case T:
                    minoType = 5;
                    break;
                case Z:
                    minoType = 6;
                    break;
                default:
                    throw new IllegalArgumentException("Invalid Mino type: " + potentialAction.getType());
            }
        } catch (IllegalArgumentException e) {
            System.err.println("[ERROR] Invalid Mino type encountered: " + e.getMessage());
            e.printStackTrace();
            System.exit(-1); // Exit in case of an invalid Mino type
        }
    
        // Populate the feature matrix
        featureMatrix.set(0, 0, baseHeight);
        featureMatrix.set(0, 1, bumpiness);
        featureMatrix.set(0, 2, emptyBelow);
        featureMatrix.set(0, 3, fullRows);
        featureMatrix.set(0, 4, minoType);
    
        return featureMatrix;
    }
    

    private int calculateBaseHeightFromGrayscale(Matrix grayscaleMatrix) {
        for (int x = 0; x < grayscaleMatrix.getShape().getNumRows(); x++) {
            for (int y = 0; y < grayscaleMatrix.getShape().getNumCols(); y++) {
                if (grayscaleMatrix.get(x, y) > 0) { // Any occupied block
                    return x;
                }
            }
        }
        return grayscaleMatrix.getShape().getNumRows(); // No blocks, return max height
    }
    
    private int calculateBumpinessFromGrayscale(Matrix grayscaleMatrix) {
        int[] columnHeights = new int[grayscaleMatrix.getShape().getNumCols()];
        int bumpiness = 0;
    
        for (int y = 0; y < grayscaleMatrix.getShape().getNumCols(); y++) {
            columnHeights[y] = calculateColumnHeightFromGrayscale(grayscaleMatrix, y);
        }
    
        for (int i = 0; i < columnHeights.length - 1; i++) {
            bumpiness += Math.abs(columnHeights[i] - columnHeights[i + 1]);
        }
    
        return bumpiness;
    }
    
    private int calculateColumnHeightFromGrayscale(Matrix grayscaleMatrix, int col) {
        for (int row = 0; row < grayscaleMatrix.getShape().getNumRows(); row++) {
            if (grayscaleMatrix.get(row, col) > 0) {
                return grayscaleMatrix.getShape().getNumRows() - row;
            }
        }
        return 0; // Column is empty
    }
    
    private int calculateEmptySpacesFromGrayscale(Matrix grayscaleMatrix) {
        int emptySpaces = 0;
        for (int col = 0; col < grayscaleMatrix.getShape().getNumCols(); col++) {
            boolean blockFound = false;
            for (int row = 0; row < grayscaleMatrix.getShape().getNumRows(); row++) {
                if (grayscaleMatrix.get(row, col) > 0) {
                    blockFound = true;
                } else if (blockFound) {
                    emptySpaces++;
                }
            }
        }
        return emptySpaces;
    }

    
    
    private int calculateFullRowsFromGrayscale(Matrix grayscaleMatrix) {
        int fullRows = 0;
        for (int row = 0; row < grayscaleMatrix.getShape().getNumRows(); row++) {
            boolean isFull = true;
            for (int col = 0; col < grayscaleMatrix.getShape().getNumCols(); col++) {
                if (grayscaleMatrix.get(row, col) == 0) {
                    isFull = false;
                    break;
                }
            }
            if (isFull) {
                fullRows++;
            }
        }
        return fullRows;
    }
    
    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */

     @Override
     public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
         int currentGame = (int) gameCounter.getCurrentGameIdx(); // Current game index
         int currentTurn = (int) gameCounter.getCurrentMoveIdx(); // Current move index in the game
     
         // Exploration probability settings
         double minExplorationRate = 0.05; // The lowest exploration rate allowed
         double startingExplorationRate = 1.0; // Starting exploration rate
         double decayFactor = 0.99 - (currentGame * 0.001); // Adjusts how quickly exploration decays
     
         // Calculate the current exploration rate based on decay and the current turn
         double adjustedExplorationRate = Math.max(
             minExplorationRate,
             startingExplorationRate - (currentTurn * (startingExplorationRate - minExplorationRate)) / decayFactor
         );
     
         // Determine whether to explore based on a random value
         return this.getRandom().nextDouble() < adjustedExplorationRate;
     }
     
     /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game) {
        List<Mino> possibleMoves = game.getFinalMinoPositions();
        int numPossibilities = possibleMoves.size();

        // Early return if no moves are available
        if (numPossibilities == 0) {
            System.err.println("[ERROR] No possible moves found!");
            return null;
        }

        // Initialize a matrix to store Q-values
        Matrix qValues = Matrix.zeros(1, numPossibilities);
        double totalQValue = 0.0;

        // Compute Q-values for all possible moves
        for (int i = 0; i < numPossibilities; i++) {
            try {
                Matrix input = this.getQFunctionInput(game, possibleMoves.get(i));
                double qValue = this.getQFunction().forward(input).get(0, 0);
                qValues.set(0, i, Math.exp(qValue)); // Exponential scaling
                totalQValue += Math.exp(qValue);
            } catch (Exception e) {
                System.err.println("[ERROR] Failed to calculate Q-value for move " + i);
                e.printStackTrace();
            }
        }

        // Normalize Q-values to probabilities
        Matrix probabilities = null;
        try {
            probabilities = qValues.ediv(Matrix.full(1, numPossibilities, totalQValue));
        } catch (Exception e) {
            System.err.println("[ERROR] Failed to normalize Q-values.");
            e.printStackTrace();
            System.exit(-1);
        }

        // Use a weighted selection based on probabilities
        double randomThreshold = this.getRandom().nextDouble();
        double cumulativeProbability = 0.0;

        for (int i = 0; i < numPossibilities; i++) {
            cumulativeProbability += probabilities.get(0, i);
            if (randomThreshold <= cumulativeProbability) {
                return possibleMoves.get(i); // Select the move based on cumulative probability
            }
        }

        // Fallback: If no move is selected, return a random move (rare case)
        //System.err.println("[WARNING] No move selected using probabilities. Defaulting to random choice.");
        return possibleMoves.get(this.getRandom().nextInt(numPossibilities));
    }

     /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    public void trainQFunction(Dataset dataset, LossFunction lossFunction, Optimizer optimizer, long numUpdates) {
        for (int epochIdx = 0; epochIdx < numUpdates; ++epochIdx) {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix>> batchIterator = dataset.iterator();

            while (batchIterator.hasNext()) {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                            lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */

    // @Override
    // public double getReward(final GameView game) {
    //     Board board = game.getBoard();
    //     double reward = 0.0;

    //     // Initialize feature variables
    //     int numSpacesBelowHighestMino = 0;
    //     int sandpaper = 0;
    //     int completedRows = 0;
    //     int linesCleared = 0;
    //     int largestYVal = Board.NUM_ROWS;
    //     int perfectClears = 0;
    //     int tetrisBonus = 0;
    //     int tSpinBonus = 0;
    //     Integer[] colMax = new Integer[Board.NUM_COLS];
    //     Coordinate highestCoord = null;

    //     // Parse through the board
    //     for (int y = 0; y < Board.NUM_ROWS; y++) {
    //         boolean isFullRow = true;

    //         for (int x = 0; x < Board.NUM_COLS; x++) {
    //             if (board.isCoordinateOccupied(x, y)) {
    //                 if (highestCoord == null) {
    //                     highestCoord = new Coordinate(x, y);
    //                 }
    //                 if (colMax[x] == null) {
    //                     colMax[x] = y;
    //                 }
    //                 isFullRow = false;
    //             } else {
    //                 if (colMax[x] != null) {
    //                     numSpacesBelowHighestMino++;
    //                 }
    //             }
    //         }

    //         if (isFullRow) {
    //             completedRows++;
    //             linesCleared++;
    //         }
    //     }

    //     if (highestCoord != null) {
    //         largestYVal = highestCoord.getYCoordinate();
    //     }

    //     for (int i = 0; i < Board.NUM_COLS - 1; i++) {
    //         int currentHeight = colMax[i] != null ? colMax[i] : Board.NUM_ROWS;
    //         int nextHeight = colMax[i + 1] != null ? colMax[i + 1] : Board.NUM_ROWS;
    //         sandpaper += Math.abs(currentHeight - nextHeight);
    //     }

    //     int pointsEarned = game.getScoreThisTurn();
    //     if (board.isClear()) {
    //         perfectClears++;
    //     }
    //     if (completedRows == 4) {
    //         tetrisBonus++;
    //     }

    //     reward = (10 * pointsEarned) // Lower weight for points
    //         + (5 * completedRows) // Reduced row clearing emphasis
    //         + (linesCleared * 0.5) // Further lower line-clearing rewards
    //         + (6 * perfectClears)
    //         - (10 * sandpaper) // Stronger penalty for uneven columns
    //         - (10 * numSpacesBelowHighestMino); // Stronger penalty for holes

    //     reward += (4 * tetrisBonus) + (6 * tSpinBonus);

    //     return reward;
    // }

    @Override
    public double getReward(final GameView game) {
        Board board = game.getBoard();
        double reward = 0.0;

        // Feature variables
        int highestBlockY = 22; // Default to maximum height
        Coordinate topBlock = null;
        boolean foundHighestBlock = false;

        int gapsBelowTopBlock = 0;
        Integer[] columnHeights = new Integer[10];

        int columnVariance = 0; // Difference in heights between adjacent columns
        int fullRowCount = 0;

        // Analyze the board to collect features
        for (int row = 0; row < 22; row++) {
            boolean rowIsFull = true;

            for (int col = 0; col < 10; col++) {
                // Check if the current cell is occupied
                if (board.isCoordinateOccupied(col, row)) {
                    // Record the highest block
                    if (!foundHighestBlock) {
                        topBlock = new Coordinate(col, row);
                        foundHighestBlock = true;
                    }

                    // Record the highest block in the column
                    if (columnHeights[col] == null) {
                        columnHeights[col] = row;
                    }
                    rowIsFull = false;
                } else { // Empty cell
                    if (columnHeights[col] != null) {
                        gapsBelowTopBlock++;
                    }
                }
            }

            // Count rows that are completely filled
            if (rowIsFull && foundHighestBlock) {
                fullRowCount++;
            }
        }

        // Determine the height of the highest block
        if (topBlock != null) {
            highestBlockY = topBlock.getYCoordinate();
        }

        // Calculate column variance (sandpaper)
        for (int i = 0; i < 9; i++) {
            int currentColHeight = columnHeights[i] != null ? columnHeights[i] : 22;
            int nextColHeight = columnHeights[i + 1] != null ? columnHeights[i + 1] : 22;
            columnVariance += Math.abs(currentColHeight - nextColHeight);
        }

        // Points earned during this turn
        int turnScore = game.getScoreThisTurn();

        // Reward calculation
        reward = (50 * turnScore) // Prioritize scoring
            + (10 * fullRowCount) // Encourage full rows
            - ((3 * columnVariance) + (5 * gapsBelowTopBlock)) / (double) highestBlockY; // Penalize uneven columns and gaps

        return reward;
    }


}
