# Use these points
top_left = (np.float64(181.78094550703716), np.float64(85.94911584265608)) 
top_right = (np.float64(514.5795423214778), np.float64(76.44058450510063)) 
bottom_left = (np.float64(68.59891703732352), np.float64(398.17540127634885)) 
bottom_right = (np.float64(635.2059086839749), np.float64(389.4583706356311))

def findGridPoints(warpedEdges, minLength, maxLineGap):
    lines, intersectPoints = findLines(warpedEdges, threshold=80, 
                                        minLength=minLength, 
                                        maxLineGap=maxLineGap, 
                                        clusterGrid=15)
    return intersectPoints

# Sort into rows and columns
def sortGridPoints(pts, tolerance=10):
    pts = pts[np.argsort(pts[:,1])] # Sort by y
    rows = []
    row = [pts[0]]
    
    for p in pts[1:]:
        if abs(p[1] - row[0][1]) < tolerance:
            row.append(p)
        else:
            rows.append(sorted(row, key=lambda p: p[0])) # Sort each row by x
            row = [p]
    rows.append(sorted(row, key=lambda p: p[0]))
    
    return rows # Rows[i][j] = (x,y) of grid point at row i, col j