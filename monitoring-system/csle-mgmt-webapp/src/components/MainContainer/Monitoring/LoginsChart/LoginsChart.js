import React from 'react';
import './LoginsChart.css';
import {
    CartesianGrid,
    Label,
    Legend, Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";

const LoginsChart = React.memo((props) => {
        const width = 500
        const height = 200
        const margin = {
            top: 10,
            right: 30,
            left: 15,
            bottom: 25
        }

        if (props.stats !== undefined && props.stats.length > 0) {
            const data = props.stats.map((host_metrics, index) => {
                return {
                    "t": (index + 1),
                    "Failed login attempts": parseInt(host_metrics.num_failed_login_attempts),
                    "Successful login attempts": parseInt(host_metrics.num_login_events)
                }
            })
            var domain = [0, Math.max(1, data.length)]

            return (
                <ResponsiveContainer width='100%' height={300}>
                    <LineChart
                        width={width}
                        height={height}
                        data={data}
                        margin={margin}
                    >
                        <CartesianGrid strokeDasharray="3 3"/>
                        <XAxis dataKey="t" type="number" domain={domain}>
                            <Label value="Time-step t" offset={-20} position="insideBottom" className="largeFont"/>
                        </XAxis>
                        <YAxis type="number">
                            <Label angle={270} value="# events" offset={0} position="insideLeft" className="largeFont"
                                   dy={50}/>
                        </YAxis>
                        <Tooltip/>
                        <Legend verticalAlign="top" wrapperStyle={{position: 'relative', fontSize: '15px'}}
                                className="largeFont"/>
                        <Line isAnimationActive={props.animation} animation={props.animation} type="monotone"
                              dataKey="Failed login attempts"
                              stroke="#8884d8" addDot={false} activeDot={{r: 8}}
                              animationEasing={'linear'}
                              animationDuration={((1 - (props.animationDuration / 100)) * props.animationDurationFactor)}/>
                        <Line animation={props.animation} type="monotone" dataKey="Successful login attempts"
                              stroke="#82ca9d" animationEasing={'linear'}
                              animationDuration={((1 - (props.animationDuration / 100)) * props.animationDurationFactor)}
                              isAnimationActive={props.animation}/>
                    </LineChart>
                </ResponsiveContainer>
            )

        } else {
            return (
                <div></div>
            )
        }
    }
)
LoginsChart.propTypes = {};

LoginsChart.defaultProps = {};

export default LoginsChart;
