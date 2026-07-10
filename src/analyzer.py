import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm
from src.common import HIGHWAY_TYPES
from src.common import EXPERIMENTS, L_PATH, U_PATH, HYBRID_ANALYSIS
from src.common import UNIFORM_RANDOM_SAMPLING, STRATIFIED_RANDOM_SAMPLING, RANDOM_SAMPLING_HIGH_TRAFFIC_ONLY, UNCERTAINTY_ONLY, FURTHEST_SAMPLING, HYBRID
from joblib import load

class Analyzer:

    def get_distribution(self, df, column_name):
        roads = df[["osm_id", column_name]].drop_duplicates(subset="osm_id")
        counts = roads[column_name].value_counts()
        distribution = counts.reset_index()
        distribution.columns = [column_name, "Count"]
        total_roads = distribution["Count"].sum()
        distribution["Percentage"] = (distribution["Count"] / total_roads * 100).round(2)
        return distribution
    
    def get_speed_distribution(self, df):
        roads = df[["osm_id", "maxspeed", "highway"]]
        roads = roads.drop_duplicates(subset="osm_id")
        total_roads = len(roads)
        table = []
        for speed in sorted(roads["maxspeed"].dropna().unique()):
            roads_at_speed = roads[roads["maxspeed"] == speed]
            count = len(roads_at_speed)
            percentage = round(count / total_roads * 100, 2)
            highway_counts = roads_at_speed["highway"].value_counts()
            most_common_highway = highway_counts.index[0]
            most_common_count = highway_counts.iloc[0]
            most_common_percentage = round(most_common_count / count * 100,2)
            row = {"Max Speed": speed,"Count": count,"Percentage": percentage,"Most Frequent Highway": most_common_highway,"Highway Count": most_common_count,"Highway Percentage": most_common_percentage}
            table.append(row)
        distribution = pd.DataFrame(table)
        return distribution
    
    def get_traffic_variation_table(self, df):
        res = []
        traffic_hours = [8, 13, 18]
        data = df.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["date"] = data["timestamp"].dt.date
        data["hour"] = data["timestamp"].dt.hour
        data["weekday"] = data["timestamp"].dt.weekday
        data = data[(data["weekday"] == 4) &(data["hour"].isin(traffic_hours))]
        highways = data["highway"].unique()
        for highway in highways:
            highway_data = data[data["highway"] == highway]
            info = {"Highway Type": highway}
            for hour in traffic_hours:
                hour_data = highway_data[highway_data["hour"] == hour]
                hourly_totals = hour_data.groupby(["osm_id", "date", "hour"])["sampleSize"].sum()
                info[f"{hour}:00"] = round(hourly_totals.mean(), 1)
            res.append(info)
        return pd.DataFrame(res)
    
    def plot_week_traffic_variation(self, df):
        osm_id = "1012809349_247718442_0"
        data = df.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data[data["osm_id"] == osm_id]
        data = data[(data["timestamp"] >= "2024-01-08") &(data["timestamp"] < "2024-01-15")]
        weekdays = data[data["timestamp"].dt.weekday < 5]
        weekend = data[data["timestamp"].dt.weekday >= 5]
        plt.figure(figsize=(8, 5))
        plt.plot(weekdays["timestamp"],weekdays["sampleSize"],color="blue",label="Weekdays")
        plt.plot(weekend["timestamp"],weekend["sampleSize"],color="orange",label="Weekend")
        plt.title(f"Evolution of traffic on osm id {osm_id}")
        plt.xlabel("Date")
        plt.ylabel("sampleSize")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_month_records_by_week(self, df):
        osm_id = "281321141_280112671_0"

        data = df.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data[data["osm_id"] == osm_id]
        data = data.sort_values("timestamp")

        start_date = data["timestamp"].dt.normalize().min()
        data["day_number"] = (data["timestamp"].dt.normalize() - start_date).dt.days
        data["week_number"] = data["day_number"] // 7
        data["hour_in_week"] = ((data["day_number"] % 7) * 24+ data["timestamp"].dt.hour+ data["timestamp"].dt.minute / 60)

        number_of_weeks = data["week_number"].max() + 1

        fig, axes = plt.subplots(number_of_weeks,1,figsize=(12, 2.2 * number_of_weeks),sharex=False,sharey=True)

        for week in range(number_of_weeks):
            ax = axes[week]
            week_data = data[data["week_number"] == week]
            ax.plot(week_data["hour_in_week"],week_data["sampleSize"])
            ax.set_title(f"Week {week + 1}", fontsize=9)
            ax.set_ylabel("sampleSize")
            ax.set_xlim(0, 168)
            ax.grid(True, alpha=0.3)
            for day in range(7):
                day_start = day * 24
                day_end = day_start + 24
                current_date = start_date + pd.Timedelta(days=week * 7 + day)
                ax.axvline(day_start, linestyle="--", linewidth=0.6, alpha=0.6)
                if current_date.weekday() >= 5:
                    ax.axvspan(day_start, day_end, color="yellow", alpha=0.2)
                day_label = current_date.strftime("%a %d")
                ax.text(day_start + 1,ax.get_ylim()[1] * 0.90,day_label,fontsize=7)
        axes[-1].set_xlabel("Time within week (hours)")
        plt.tight_layout()
        plt.show()


    def plot_highway_distribution(self, df):
        plt.figure(figsize=(9, 5))

        highways = sorted(df["highway"].unique())
        colors = sns.color_palette("tab10", n_colors=len(highways))

        for highway, color in zip(highways, colors):
            data = df[df["highway"] == highway]
            plt.scatter(data["lon"],data["lat"],s=3,color=color,label=highway)

        plt.title("Geographical Distribution of segments by highway types", fontsize=10)
        plt.xlabel("Longitude", fontsize=8)
        plt.ylabel("Latitude", fontsize=8)

        plt.xlim(3.0, 6.5)
        plt.ylim(49.5, 50.8)

        plt.legend(title="Highway Type:", fontsize=7, title_fontsize=7, loc="lower left")
        plt.tick_params(axis="both", labelsize=8)
        plt.tight_layout()
        plt.show()
    
    def plot_highway_distribution_L(self, df):
        highways = ["motorway", "motorway_link", "primary", "secondary", "tertiary", "trunk", "trunk_link"]
        color_map = dict(zip(highways, sns.color_palette("tab10", len(highways))))
        plt.figure(figsize=(9, 5))
        for highway in highways:
            plt.scatter(df.loc[df[f"highway_{highway}"] == 1, "lon"],df.loc[df[f"highway_{highway}"] == 1, "lat"],s=3,color=color_map[highway],label=highway)
        plt.title("Geographical Distribution of segments by highway types", fontsize=10)
        plt.xlabel("Longitude", fontsize=8)
        plt.ylabel("Latitude", fontsize=8)
        plt.xlim(3.0, 6.5)
        plt.ylim(49.5, 50.8)
        plt.legend(title="Highway Type:", fontsize=7, title_fontsize=7, loc="lower left")
        plt.tick_params(axis="both", labelsize=8)
        plt.tight_layout()
    plt.show()

    def get_highway_distribution_table(self, df):
        counts = df[HIGHWAY_TYPES].sum()
        total = counts.sum()
        table = pd.DataFrame({"Highway type": HIGHWAY_TYPES,"Count": counts.values,"Percentage": (counts / total * 100).round(2)})
        table.loc[len(table)] = ["Total", total, 100.00]
        return table
    
    def plot_evolution_accuracy_per_budget(self, summary):
        summary = summary.sort_values("budget")
        plots = [("avg_r2", "std_r2", "$R^2$", "Evolution of $R^2$ per budget", (0.5, 0.9)),("avg_mse", "std_mse", "MSE", "Evolution of MSE per budget", (10, 100)),("avg_mae_mr", "std_mae_mr", "MAE_MR", "Evolution of MAE_MR per budget", (0, 6))]
        for mean_col, std_col, ylabel, title, ylim in plots:
            x = summary["budget"]
            y = summary[mean_col]
            std = summary[std_col]
            plt.figure(figsize=(8, 5))
            plt.plot(x,y,marker="o",linewidth=2 )
            plt.fill_between(x,y - std,y + std,alpha=0.2 )
            plt.xlabel("Budget")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xlim(0, 550)
            plt.ylim(ylim)
            plt.xticks([0, 100, 200, 300, 400, 500])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def print_evolution_accuracy_per_budget(self, summary):
        table = summary.sort_values("budget")
        table = table[["budget","avg_r2", "std_r2","avg_mse", "std_mse","avg_mae_mr", "std_mae_mr"]]
        print(table.to_string(index=False))
    
    def plot_per_category(self, mean_mse_category, std_mse_category):
        plt.figure(figsize=(10, 5))
        colors = plt.get_cmap("Set3").colors
        plt.bar(HIGHWAY_TYPES,mean_mse_category,yerr=std_mse_category,capsize=5, color=colors[:len(HIGHWAY_TYPES)])
        plt.xlabel("Highway type")
        plt.ylabel("MSE")
        plt.title("MSE per highway type (log scale)")
        plt.xticks(rotation=45, ha="right")
        plt.yscale("log")
        plt.tight_layout()
        plt.show()
    
    def plot_strategy_comparison(self):
        strategy_names = {UNIFORM_RANDOM_SAMPLING: "Uniform random", STRATIFIED_RANDOM_SAMPLING: "Stratified random", RANDOM_SAMPLING_HIGH_TRAFFIC_ONLY: "Higher Traffic Random", UNCERTAINTY_ONLY: "Uncertainty sampling", FURTHEST_SAMPLING: "Furthest point sampling", HYBRID: "Hybrid uncertainty-diversity"}
        strategies = {}
        for strategy_id, strategy_name in strategy_names.items():
            path = f"{EXPERIMENTS}/dps_full_{strategy_id}.joblib"
            df = load(path)
            summary = df.groupby("budget").agg(avg_r2=("r2", "mean"), avg_mae_mr=("mae_mr", "mean")).reset_index()
            strategies[strategy_name] = summary
        plots = [("avg_r2", "$R^2$", "Comparison of $R^2$ per budget", (0.50, 0.85)), ("avg_mae_mr", "MAE_MR", "Comparison of MAE_MR per budget", (1.5, 6))]
        for metric, ylabel, title, ylim in plots:
            plt.figure(figsize=(8, 5))
            for strategy_name, summary in strategies.items():
                summary = summary.sort_values("budget")
                linewidth = 3 if strategy_name == "Hybrid uncertainty-diversity" else 2
                plt.plot(summary["budget"], summary[metric], marker="o", linewidth=linewidth, label=strategy_name)
            plt.xlabel("Budget")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xlim(0, 550)
            plt.ylim(ylim)
            plt.xticks([0, 100, 200, 300, 400, 500])
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    def plot_road_type_by_batch(self, data_hybrid, seeds):
        road_types = [col.removeprefix("highway_") for col in HIGHWAY_TYPES]
        counts = (data_hybrid.groupby(["seed", "batch", "road_type"]).size().reset_index(name="count"))
        rows = []
        for seed in seeds:
            for batch in sorted(data_hybrid["batch"].unique()):
                batch_counts = counts[(counts["seed"] == seed) &(counts["batch"] == batch)]
                for road_type in road_types:
                    match = batch_counts[batch_counts["road_type"] == road_type]
                    count = 0 if match.empty else match["count"].iloc[0]
                    rows.append({"seed": seed, "batch": batch,"road_type": road_type,"count": count})
        counts = pd.DataFrame(rows)
        counts["percentage"] = counts.groupby(["seed", "batch"])["count"].transform(lambda x: 100 * x / x.sum())
        summary = counts.groupby(["batch", "road_type"])["percentage"].mean().reset_index()
        mean_table = summary.pivot(index="batch", columns="road_type", values="percentage").fillna(0)
        ax = mean_table.plot(kind="bar", stacked=True, figsize=(12, 5))
        ax.set_ylim(0, 105)
        ax.set_ylabel("Average percentage across seeds")
        ax.set_xlabel("Batch")
        ax.set_title("Average road type composition per batch")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()
    
    def compare_with_full_dataset(self, data_hybrid):
        L = load(L_PATH)
        U = load(U_PATH)
        full_df = pd.concat([L[HIGHWAY_TYPES], U[HIGHWAY_TYPES]], ignore_index=True)
        acquisition_df = data_hybrid[HIGHWAY_TYPES]
        final_labelled_df = pd.concat([L[HIGHWAY_TYPES], data_hybrid[HIGHWAY_TYPES]], ignore_index=True)
        full_percent = 100 * full_df.sum() / full_df.sum().sum()
        acquisition_percent = 100 * acquisition_df.sum() / acquisition_df.sum().sum()
        final_percent = 100 * final_labelled_df.sum() / final_labelled_df.sum().sum()
        comparison = pd.DataFrame({"full_dataset_%": full_percent,"al_acquisition_%": acquisition_percent,"final_labelled_%": final_percent,})
        comparison.index = comparison.index.str.replace("highway_", "")
        comparison = comparison.sort_values("full_dataset_%", ascending=False)
        print(comparison.round(2))
        return comparison
    
    def compare_seed_stability(self, data_hybrid):
        seed_counts = data_hybrid.groupby("seed")[HIGHWAY_TYPES].sum()
        seed_percent = seed_counts.div(seed_counts.sum(axis=1), axis=0) * 100
        mean_percent = seed_percent.mean(axis=0)
        std_percent = seed_percent.std(axis=0)
        stability = pd.DataFrame({"mean_%": mean_percent,"std_%": std_percent})
        stability.index = stability.index.str.replace("highway_", "")
        stability = stability.sort_values("mean_%", ascending=False)
        print(stability.round(2))
        return stability
    
    def plot_seed_geography(self, batches, seed=1):
        all_points = []
        for batch_id, df in enumerate(batches):
            geography = df[["lon", "lat"]].copy()
            geography["batch"] = batch_id
            all_points.append(geography)
        points = pd.concat(all_points, ignore_index=True)
        n_batches = points["batch"].nunique()
        cmap = get_cmap("coolwarm", n_batches)
        norm = BoundaryNorm(range(n_batches + 1), cmap.N)
        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(points["lon"],points["lat"],c=points["batch"],cmap=cmap,norm=norm,s=14,alpha=0.8)
        cbar = plt.colorbar(scatter, ticks=range(n_batches))
        cbar.set_label("Batch")
        cbar.ax.set_yticklabels(range(n_batches))
        plt.xlim(points["lon"].min() - 0.1, points["lon"].max() + 0.1)
        plt.ylim(points["lat"].min() - 0.1, points["lat"].max() + 0.1)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Geographical distribution of selected points by batch (seed {seed})")
        plt.show()
    
    def plot_seed_geography_with_labelled(self, batches, L, seed=1):
        all_points = []
        for batch_id, df in enumerate(batches):
            geography = df[["lon", "lat"]].copy()
            geography["batch"] = batch_id
            all_points.append(geography)
        points = pd.concat(all_points, ignore_index=True)
        n_batches = points["batch"].nunique()
        cmap = get_cmap("coolwarm", n_batches)
        norm = BoundaryNorm(range(n_batches + 1), cmap.N)

        plt.figure(figsize=(11, 7))
        scatter = plt.scatter( points["lon"], points["lat"], c=points["batch"], cmap=cmap,
            norm=norm, s=18, alpha=0.8)
        plt.scatter(L["lon"],L["lat"],color="green",s=20,alpha=0.5,edgecolors="black",linewidths=0.15,label="Initial labelled set")
        cbar = plt.colorbar(scatter, ticks=range(n_batches))
        cbar.set_label("Batch")
        cbar.ax.set_yticklabels(range(n_batches))
        plt.xlim(points["lon"].min() - 0.1, points["lon"].max() + 0.1)
        plt.ylim(points["lat"].min() - 0.1, points["lat"].max() + 0.1)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Geographical distribution of selected points and initial labelled set (seed {seed})")
        plt.legend()
        plt.show()
    
    def plot_seed_geography_first_batches(self, batches, L, seed=1, max_batch=4):
        all_points = []
        for batch_id, df in enumerate(batches):
            if batch_id >= max_batch:
                break
            geography = df[["lon", "lat"]].copy()
            geography["batch"] = batch_id
            all_points.append(geography)
        points = pd.concat(all_points, ignore_index=True)
        cmap = get_cmap("coolwarm")
        norm = plt.Normalize(vmin=0, vmax=20)
        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(points["lon"],points["lat"],c=points["batch"],cmap=cmap,norm=norm,s=18,alpha=0.85)
        plt.scatter(L["lon"],L["lat"],color="green",s=20,alpha=0.5,edgecolors="black",linewidths=0.15,label="Initial labelled set")
        cbar = plt.colorbar(scatter, ticks=range(max_batch))
        cbar.set_label("Batch")
        cbar.ax.set_yticklabels(range(max_batch))
        plt.xlim(points["lon"].min() - 0.1, points["lon"].max() + 0.1)
        plt.ylim(points["lat"].min() - 0.1, points["lat"].max() + 0.1)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Geographical distribution of the first batches (seed {seed})")
        plt.legend()
        plt.show()
    
    def analyze_recurrent_points(self, data_hybrid):
        unique_points = data_hybrid[["lon", "lat", "seed"]].drop_duplicates()
        recurrence = (unique_points.groupby(["lon", "lat"]).agg(n_seeds=("seed", "nunique")).reset_index())
        summary = (recurrence["n_seeds"].value_counts().sort_index().reset_index())
        summary.columns = ["n_seeds", "n_points"]
        total_unique = len(recurrence)
        summary["percentage"] = (100 * summary["n_points"] / total_unique)
        print(summary.round(2))
        return summary, recurrence
    
    def analyze_batch_recurrence(self, data_hybrid):
        unique_points = data_hybrid[["lon", "lat", "seed", "batch"]].drop_duplicates()
        recurrence = (unique_points.groupby(["lon", "lat"]).agg(n_seeds=("seed", "nunique"),mean_batch=("batch", "mean"),std_batch=("batch", "std"),min_batch=("batch", "min"),max_batch=("batch", "max")).reset_index())
        summary = (recurrence.groupby("n_seeds").agg(n_points=("n_seeds", "count"),mean_batch=("mean_batch", "mean"),std_batch=("mean_batch", "std"),avg_min_batch=("min_batch", "mean"),avg_max_batch=("max_batch", "mean")).reset_index() )
        summary["percentage"] = 100 * summary["n_points"] / summary["n_points"].sum()
        print(summary.round(2))
        return summary, recurrence
    
    def plot_points_present_in_all_seeds(self, data_hybrid, seeds):
        unique_points = data_hybrid[["lon", "lat", "seed", "road_type"]].drop_duplicates()
        recurrence = (unique_points.groupby(["lon", "lat"]).agg(n_seeds=("seed", "nunique"),road_type=("road_type", "first")).reset_index())
        recurrent_points = recurrence[recurrence["n_seeds"] == len(seeds)]
        plt.figure(figsize=(11, 7))
        for road_type, group in recurrent_points.groupby("road_type"):
            plt.scatter(group["lon"],group["lat"],s=25,alpha=0.8,label=road_type )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Datapoints selected in all seeds by road type")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()
    
    def plot_batch_zero_scores(self, seed):
        diversity = load(f"{HYBRID_ANALYSIS}/diversity_batch_0.joblib")
        uncertainty = load(f"{HYBRID_ANALYSIS}/uncertainty_batch_0.joblib")
        U = load(f"{HYBRID_ANALYSIS}/U_batch_0.joblib")
        L = load(L_PATH)
        points = U[["lon", "lat"]].copy()
        points["diversity"] = diversity
        points["uncertainty"] = uncertainty

        # Diversity map
        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(points["lon"],points["lat"],c=points["diversity"],cmap="coolwarm",s=10,alpha=0.8)
        plt.scatter( L["lon"], L["lat"], color="green", s=20, alpha=0.5, edgecolors="black", linewidths=0.15, label="Initial labelled set")
        plt.colorbar(scatter, label="Diversity score")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Diversity score distribution at batch 0 (seed {seed})")
        plt.legend()
        plt.show()

        # Uncertainty map
        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(points["lon"],points["lat"],c=points["uncertainty"],cmap="coolwarm",s=10,alpha=0.8)
        plt.colorbar(scatter, label="Uncertainty score")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Uncertainty score distribution at batch 0 (seed {seed})")
        plt.show()

        hybrid = load(f"{HYBRID_ANALYSIS}/hybrid_batch_0.joblib")
        points["hybrid"] = hybrid
        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(points["lon"],points["lat"],c=points["hybrid"],cmap="coolwarm",s=10,alpha=0.8)
        plt.colorbar(scatter, label="Hybrid score")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Hybrid score distribution at batch 0 (seed {seed})")
        plt.show()
    