import numpy as np

def generate_random_geografic_points(n, xlim=(-90, 90), ylim=(-180, 180)):
    """
    Gera n_points pontos geográficos aleatórios (latitude, longitude).
    
    Parâmetros:
        n_points (int): número de pontos a gerar
        lat_range (tuple): intervalo para latitude (padrão: -90 a 90)
        lon_range (tuple): intervalo para longitude (padrão: -180 a 180)
            relativo ao globo mundial
        
    Retorno:
        np.ndarray: matriz (n_points, 2) com colunas [latitude, longitude]
    """
    # latitudes = np.random.uniform(low=lat_range[0], high=lat_range[1], size=n_points)
    # longitudes = np.random.uniform(low=lon_range[0], high=lon_range[1], size=n_points)
    # return np.stack((latitudes, longitudes), axis=1)

    rng = np.random.default_rng()
    xs = rng.uniform(xlim[0], xlim[1])
    ys = rng.uniform(ylim[0], ylim[1])
    return (np.float64(xs), np.float64(ys))


class Distances:

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # km
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    @staticmethod
    def euclidean(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def manhattan(x1, y1, x2, y2):
        return np.abs(x1 - x2) + np.abs(y1 - y2)

if __name__ =="__main__":
    # Exemplo de uso:
    n_points = 20
    pontos = generate_random_geografic_points(n_points)
    print(pontos)
    print(pontos[0])
    print(f"Distancia (Haversine): {Distances.haversine(pontos[0][0], pontos[0][1], pontos[1][0],pontos[1][0])}")