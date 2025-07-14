/************************************************************************
 *
 * (c) 2025 Machines With Vision Ltd.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Machines With Vision Limited and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Machines With Vision Limited
 * and its suppliers and may be covered by U.K. and Foreign Patents,
 * patents in process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from Machines With Vision Limited.
 *
 ************************************************************************/

#ifndef AKAZETIMING_H
#define AKAZETIMING_H

namespace mwvcv
{
    struct AkazeTiming
    {
        AkazeTiming()
        {
            kcontrast   = 0.0;
            scale       = 0.0;
            derivatives = 0.0;
            detector    = 0.0;
            extrema     = 0.0;
            subpixel    = 0.0;
            descriptor  = 0.0;
        }

        double kcontrast;   ///< Contrast factor computation time in ms
        double scale;       ///< Nonlinear scale space computation time in ms
        double derivatives; ///< Multiscale derivatives computation time in ms
        double detector;    ///< Feature detector computation time in ms
        double extrema;     ///< Scale space extrema computation time in ms
        double subpixel;    ///< Subpixel refinement computation time in ms
        double descriptor;  ///< Descriptors computation time in ms
    };

} // namespace mwvcv

#endif // AKAZETIMING_H
