#include <opencv2/opencv.hpp> 
#include <stdio.h> 
#include <vector>
#include <queue>
#include <limits.h>
using namespace cv; 
using namespace std;

Mat imageBig;   // Input image
Mat image;      // Input image scaled down, for quicker processing
Mat display;    // Input image copy for drawing on and displaying

vector<Point> foregroundSeed;
vector<Point> backgroundSeed;
bool collectingForeground = false;
bool collectingBackground = false;

Point prevPoint;
float factor;   // Scale factor

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
// Draw mouse movement on displayed image and collect seeds
{
    Point pt = Point(x,y);
    Point smallpt = Point(x / factor,y / factor);
    if  ( event == EVENT_LBUTTONDOWN )
    {
        if (smallpt.x >= 0 && smallpt.y >= 0 && smallpt.x < image.cols && smallpt.y < image.rows)
            foregroundSeed.push_back(smallpt);
        collectingForeground = true;
        prevPoint = pt;
    }
    else if  ( event == EVENT_RBUTTONDOWN )
    {
        if (smallpt.x >= 0 && smallpt.y >= 0 && smallpt.x < image.cols && smallpt.y < image.rows)
            backgroundSeed.push_back(smallpt);
        collectingBackground = true;
        prevPoint = pt;
    }
    else if  ( event == EVENT_LBUTTONUP )
    {
        collectingForeground = false;
    }
    else if  ( event == EVENT_RBUTTONUP )
    {
        collectingBackground = false;
    }
    else if ( event == EVENT_MOUSEMOVE )
    {
        if (collectingForeground)
        {
            if (smallpt.x >= 0 && smallpt.y >= 0 && smallpt.x < image.cols && smallpt.y < image.rows)
                foregroundSeed.push_back(smallpt);      // Add to foreground seeds
            line(display, prevPoint, pt, Scalar(255,255,0), 3);
            imshow("Image", display);
            prevPoint = pt;
        }
        else if (collectingBackground)
        {
            if (smallpt.x >= 0 && smallpt.y >= 0 && smallpt.x < image.cols && smallpt.y < image.rows)
                backgroundSeed.push_back(smallpt);      // Add to background seeds
            line(display, prevPoint, pt, Scalar(0,0,255), 3);
            imshow("Image", display);
            prevPoint = pt;
        }
    }
}

Point idx2pt(int idx)
// Convert graph index to coordinate on small image
{
    idx--;
    int x = idx % image.cols;
    int y = idx / image.cols;
    return Point(x, y);
}

int pt2idx(Point pt)
// Convert coordinate on small image to graph index
{
    return pt.y * image.cols + pt.x + 1;
}

Point idx2pt_display(int idx)
// Convert graph index to coordinate on big image
{
    idx--;
    int x = (idx % image.cols) * factor;
    int y = (idx / image.cols) * factor;
    return Point(x, y);
}

int pt_display2idx(Point pt)
// Convert coordinate on big image to graph index
{
    return int(pt.y / factor) * image.cols + int(pt.x / factor) + 1;
}

// sigma = 30
double two_sigma_sqare = 1800;

int calculatePenalty (int ip, int iq)
// Capacity of one channel
// If pixel difference is less than sigma, the capacity is big
// Otherwise it's very small
{
    return int(
        100 * exp(
            double(
                -pow(ip - iq, 2)
            ) 
            / two_sigma_sqare
        )
    );
}

int getCapacity (int idx1, int idx2)
// Calculate capacity of the edge between two pixels
// All three channels are considered
{
    Vec3b color1 = image.at<Vec3b>(idx2pt(idx1));
    Vec3b color2 = image.at<Vec3b>(idx2pt(idx2));
    int val =calculatePenalty(color1[0], color2[0])
        + calculatePenalty(color1[1], color2[1])
        + calculatePenalty(color1[2], color2[2]);
    return val == 0 ? 1 : val;
}

vector<int> getNeighbour (int idx)
// Return neighbouring points
{
    vector<int> neighbour;
    Point pt = idx2pt(idx);
    if (pt.x > 0)
        neighbour.push_back(pt2idx(Point(pt.x - 1, pt.y)));
    if (pt.x < image.cols - 1)
        neighbour.push_back(pt2idx(Point(pt.x + 1, pt.y)));
    if (pt.y > 0)
        neighbour.push_back(pt2idx(Point(pt.x, pt.y - 1)));
    if (pt.y < image.rows - 1)
        neighbour.push_back(pt2idx(Point(pt.x, pt.y + 1)));
    return neighbour;
}

vector<vector<int>> capacity;   // Residual capacity matrix
vector<vector<int>> adj;        // Adj of graph
int dist;                       // Index of dist node (source node is 0)

void img2adj ()
// Convert small image to graph
{
    dist = image.rows * image.cols + 1;
    int size = dist + 1;
    capacity.resize(size, vector<int>(size));
    adj.resize(size, vector<int>(0));
    // All foreground seeds are connected to source with infinite capacity
    for (auto fgs : foregroundSeed)
    {  
        int idx = pt2idx(fgs);
        adj[0].push_back(idx);
        adj[idx].push_back(0);
        capacity[0][idx] = INT_MAX;
        capacity[idx][0] = INT_MAX;
    }
    // All pixel nodes are connected to neighbours with capacity
    // calculated based on pixel differences
    for (int i = 1; i < dist; i++) 
    {
        vector<int> neighbour = getNeighbour(i);
        for (int t : neighbour) 
        {
            adj[i].push_back(t);
            adj[t].push_back(i);
            int c = getCapacity(i, t);
            capacity[i][t] = c;
            capacity[t][i] = c;
        }
    }
    // All background seeds are connected to dist with infinite capacity
    for (auto bgs : backgroundSeed)
    {
        int s = pt2idx(bgs);
        adj[s].push_back(dist);
        adj[dist].push_back(s);
        capacity[s][dist] = INT_MAX;
        capacity[dist][s] = INT_MAX;
    }
}

int bfs(int s, int t, vector<int>& parent)
// Find a capable flow with bfs, Edmonds-Karp algorithm
{
    fill(parent.begin(), parent.end(), -1);
    parent[s] = -2;
    queue<pair<int, int>> q;
    q.push({s, INT_MAX});
    while (!q.empty()) 
    {
        int cur = q.front().first;
        int flow = q.front().second;
        q.pop();

        for (int next : adj[cur]) 
        {
            if (parent[next] == -1 && capacity[cur][next]) 
            {
                parent[next] = cur;
                int new_flow = min(flow, capacity[cur][next]);
                if (next == t)
                    return new_flow;
                q.push({next, new_flow});
            }
        }
    }
    return 0;
}

int maxflow(int s, int t) 
// Find maximum flow, Ford-Fulkerson method
{
    int flow = 0;
    vector<int> parent(dist + 1);
    int new_flow;
    while (new_flow = bfs(s, t, parent)) {
        flow += new_flow;
        int cur = parent[t];
        while (cur != s) 
        {
            int prev = parent[cur];
            if (prev != s)
            {
                // Update residual capacity with new capable flow
                capacity[prev][cur] -= new_flow;
                capacity[cur][prev] += new_flow;
            }
            cur = prev;
        }
    }
    return flow;
}

set<int> fgset;
set<int> bgset;

void displayCut()
// Display the found minimum cut on the graph
{
    queue<int> q;
    for (auto fgs : foregroundSeed)
    {
        q.push(pt2idx(fgs));
    }
    while (!q.empty())
    {
        int i = q.front();
        q.pop();
        if (fgset.find(i) != fgset.end())
            continue;
        display.at<Vec3b>(idx2pt_display(i)) = Vec3b(255,255,0);
        fgset.insert(i);
        vector<int> neighbour = getNeighbour(i);
        for (int t : neighbour) 
        {
            if (capacity[i][t] != 0 && fgset.find(t) == fgset.end())
            {
                q.push(t);
            }   
        }
    }
    for (int i = 1; i < dist; i++)
    {
        if (fgset.find(i) == fgset.end())
        {
            display.at<Vec3b>(idx2pt_display(i)) = Vec3b(0,0,255);
            bgset.insert(i);
        }
    }
}

Mat fgm, bgm;

void cutImage() 
// Cut image into foreground image and background image
{
    for (int y = 0; y < imageBig.cols; y++)
    {
        for (int x = 0; x < imageBig.rows; x++)
        {
            Point pt(x, y);
            Vec3b color = imageBig.at<Vec3b>(pt);
            int b = color[0];
            int g = color[1];
            int r = color[2];
            int i = pt_display2idx(Point(x,y));
            if (fgset.find(i) != fgset.end())
            {
                fgm.at<Vec4b>(pt) = Vec4b(b,g,r,255);
            } else {
                bgm.at<Vec4b>(pt) = Vec4b(b,g,r,255);
            }
        }
    }
}


int main(int argc, char** argv) 
{ 
    if (argc != 2) { 
        printf("usage: cut <Image_Path>\n"); 
        return -1; 
    } 
    imageBig = imread(argv[1], 1); 
    if (!imageBig.data) { 
        printf("No image data \n"); 
        return -1; 
    } 
    imageBig.copyTo(display);
    namedWindow("Image", WINDOW_AUTOSIZE); 
    setMouseCallback("Image", CallBackFunc, NULL);
    cout << "Size of input image" << endl;
    cout << "height:\t" << imageBig.rows << endl << "width:\t" << imageBig.cols << endl;
    imshow("Image", display); 
    cout << "------------------------------" << endl;
    // I can only run the program on 150 * 150 image,
    // Otherwise it get killed
    int sizeLimit = 150;
    factor = imageBig.rows > imageBig.cols ? 
        float(imageBig.rows) / sizeLimit : float(imageBig.cols) / sizeLimit;
    cout << "Image is scaled down with factor " << factor << endl;
    resize(imageBig, image, Size(int(imageBig.rows / factor), int(imageBig.cols / factor)), INTER_LINEAR);
    cout << "Size of processing image" << endl;
    cout << "height:\t" << image.rows << endl << "width:\t" << image.cols << endl;
    cout << "------------------------------" << endl;
    cout << "Please use left mouse button to seed foreground, " << endl;
    cout << "and right mouse button to seed background. " << endl;
    cout << "Press p to process. " << endl;
    while(true)
    {
        if (waitKey(1) == 'p') break;
    }
    imwrite("seed.png", display);
    cout << "------------------------------" << endl;
    cout << "Converting image to graph ..." << endl;
    img2adj();
    cout << "------------------------------" << endl;
    cout << "Calculating max flow ..." << endl;
    int flow = maxflow(0, dist);
    cout  << "Found max flow size: " << flow << endl;
    cout << "------------------------------" << endl;
    cout << "Finding minimum cut ..." << endl;
    displayCut();
    imshow("Image", display);
    imwrite("cut.png", display);
    cout << "Minimum cut displayed. " << endl;
    cout << "------------------------------" << endl;
    cout << "Cutting image ..." << endl;
    fgm = Mat(imageBig.cols, imageBig.rows, CV_8UC4, Scalar(0,0,0,0));
    bgm = Mat(imageBig.cols, imageBig.rows, CV_8UC4, Scalar(0,0,0,0));
    cutImage();
    imshow("Foreground", fgm);
    imshow("Background", bgm);
    imwrite("fg.png", fgm);
    imwrite("bg.png", bgm);
    cout << "Done." << endl;
    cout << "Press any key to exit." << endl;
    waitKey(0);
    return 0; 
}
